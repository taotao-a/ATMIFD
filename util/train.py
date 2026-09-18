import logging
import os
import time
import copy
import inspect
import torch
import torch.nn as nn
from adabelief_pytorch import AdaBelief

from tqdm import tqdm
import util.util as util


def class_weights_from_counts(normal_count, anomaly_count, cap):
    """Return the paper's cost-sensitive CE weights from labeled train counts."""
    if normal_count <= 0 or anomaly_count <= 0:
        return 1.0, 1.0
    return 1.0, min(float(normal_count) / (float(anomaly_count) + 1e-6), float(cap))


class Base(nn.Module):
    def __init__(self, model, **args):
        super(Base, self).__init__()

        self.model = model
        self.args = args  # keep a copy for ablation switches

        requested = args.get('device', 'auto')
        if requested == 'cuda' and not torch.cuda.is_available():
            raise ValueError('--device cuda requested, but CUDA is unavailable; use --device cpu or auto')
        self.use_gpu = requested == 'cuda' or (requested == 'auto' and args['gpu'] and torch.cuda.is_available())
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.batch_size = args['batch_size']

        # Training
        self.epoches = args['epochs']
        self.learning_rate = args['learning_rate']
        self.weight_decay = args['weight_decay']
        self.patience = args['patience']  # > 0: use early stop
        self.model_save_dir = args['result_dir']
        self.learning_change = args['learning_change']
        self.learning_gamma = args['learning_gamma']
        self.rec_down = args['rec_down']
        self.para_low = args['para_low']

        # Threshold search settings
        self.threshold = args.get('threshold', None)
        self.thr_search = args.get('thr_search', False)
        self.thr_min = args.get('thr_min', 0.0)
        self.thr_max = args.get('thr_max', 1.0)
        self.thr_steps = args.get('thr_steps', 200)

        if not args['evaluate']:
            logging.info('model : init weight')
            self.init_weight()

        if self.use_gpu:
            logging.info("Using GPU...")
            torch.cuda.empty_cache()
        else:
            logging.info("Using CPU...")
        self.model.to(self.device)

    # Model init
    def init_weight(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # Move tensors without forcing CUDA or retaining unnecessary input gradients.
    def input2device(self, batch_input, use_gpu=None):
        if isinstance(batch_input, dict):
            return {name: self.input2device(value) for name, value in batch_input.items()}
        if isinstance(batch_input, (tuple, list)):
            return type(batch_input)(self.input2device(value) for value in batch_input)
        value = torch.as_tensor(batch_input, dtype=torch.float32, device=self.device)
        return torch.where(torch.isnan(value), torch.zeros_like(value), value)

    # Loading model params
    def load_model(self, model_save_file="", name='loss'):
        if model_save_file == ' ':
            logging.info(f'No {self.model.name} state file')
        else:
            logging.info(f'{self.model.name} on {model_save_file} loading...')
            checkpoint = os.path.join(model_save_file, f"{self.model.name}_{name}_stage.ckpt")
            if not os.path.isfile(checkpoint):
                raise FileNotFoundError(f'Missing checkpoint: {checkpoint}')
            options = {'map_location': self.device}
            if 'weights_only' in inspect.signature(torch.load).parameters:
                options['weights_only'] = True
            # PyTorch 1.12 has no weights-only loader; only load trusted checkpoints.
            self.model.load_state_dict(torch.load(checkpoint, **options))

    # Saving model params
    def save_model(self, best_dict, model_save_dir="", name='loss'):
        file_status = os.path.join(model_save_dir, f"{self.model.name}_{name}_stage.ckpt")
        if best_dict['state'] is None:
            logging.info(f'No {self.model.name} - {name} state file')
        else:
            logging.info(f'{self.model.name} - {name}  best score:{best_dict["score"]} at epoch {best_dict["epoch"]}')
            torch.save(best_dict['state'], file_status)


class MY(Base):
    def __init__(self, model, **args):
        super().__init__(model, **args)

    # --------------------------
    # Window-level helpers
    # --------------------------
    def _window_label(self, y_real: torch.Tensor) -> torch.Tensor:
        """Aggregate node-level label to window-level label (any node anomalous => window anomalous)."""
        if y_real.dim() == 3 and y_real.size(-1) == 2:
            y_win = y_real[:, :, 1].max(dim=1).values
        elif y_real.dim() == 3 and y_real.size(-1) == 1:
            y_win = y_real.squeeze(-1).max(dim=1).values
        elif y_real.dim() == 2:
            y_win = y_real.max(dim=1).values
        elif y_real.dim() == 1:
            y_win = y_real
        else:
            raise RuntimeError(f"Unexpected y shape: {tuple(y_real.shape)}")
        return (y_win > 0.5).long()

    def _window_score(self, prob: torch.Tensor) -> torch.Tensor:
        """Aggregate node-level anomaly prob to window-level score."""
        agg = self.args.get('win_agg', 'max')
        topk = int(self.args.get('topk', 3))
        tau = float(self.args.get('lse_tau', 0.2))
        eps = 1e-6

        if prob.dim() == 2 and prob.size(-1) == 2:
            # already window-level: (B,2)
            return prob[:, 1].clamp(eps, 1 - eps)

        if prob.dim() == 3 and prob.size(-1) == 2:
            p = prob[:, :, 1].clamp(eps, 1 - eps)  # (B,N)
            if agg == 'max':
                return p.max(dim=1).values
            if agg == 'topk':
                k = min(topk, p.size(1))
                return torch.topk(p, k, dim=1).values.mean(dim=1)
            if agg == 'lse':
                # smooth-max on logits
                logit = torch.log(p) - torch.log(1 - p)
                s = tau * torch.logsumexp(logit / tau, dim=1)
                return torch.sigmoid(s)

        raise RuntimeError(f"Unexpected prob shape: {tuple(prob.shape)}")

    def _fix_prob_shape(self, cls_prob: torch.Tensor, B: int, N: int, real_B: int) -> torch.Tensor:
        """Make evaluate output compatible: (B,N,2) or (B,2)."""
        if cls_prob.dim() == 3 and cls_prob.size(-1) == 2:
            return cls_prob[:real_B]

        if cls_prob.dim() == 2 and cls_prob.size(-1) == 2:
            out_rows = cls_prob.size(0)
            if out_rows == B:
                return cls_prob[:real_B]
            if out_rows == B * N:
                return cls_prob.view(B, N, 2)[:real_B]
            raise RuntimeError(f"Unexpected cls_prob shape {tuple(cls_prob.shape)} with B={B}, N={N}")

        raise RuntimeError(f"Unexpected cls_prob shape {tuple(cls_prob.shape)}")

    # --------------------------
    # Threshold search (window-level)
    # --------------------------
    def search_threshold(self, val_loader):
        """Search best threshold on validation set to maximize window-level F1."""
        self.model.eval()
        with torch.no_grad():
            probs_list, label_list = [], []
            B = self.batch_size

            for batch_input in tqdm(val_loader):
                batch_input = self.input2device(batch_input, self.use_gpu)

                # unpack
                if isinstance(batch_input, dict):
                    x_node = batch_input['data_node']
                    y = batch_input['groundtruth_real']
                    real_B = x_node.size(0)
                    N = x_node.size(2)

                    # pad
                    if real_B < B:
                        pad_B = B - real_B
                        for k in batch_input:
                            t = batch_input[k]
                            batch_input[k] = torch.cat([t, t.new_zeros((pad_B, *t.shape[1:]))], dim=0)
                else:
                    x_node = batch_input[0]
                    y = batch_input[-1]
                    real_B = x_node.size(0)
                    N = x_node.size(2)
                    if real_B < B:
                        pad_B = B - real_B
                        x_node = torch.cat([x_node, x_node.new_zeros((pad_B, *x_node.shape[1:]))], dim=0)
                        x_edge = torch.cat([batch_input[1], batch_input[1].new_zeros((pad_B, *batch_input[1].shape[1:]))], dim=0)
                        x_log = torch.cat([batch_input[2], batch_input[2].new_zeros((pad_B, *batch_input[2].shape[1:]))], dim=0)
                        y = torch.cat([y, y.new_zeros((pad_B, *y.shape[1:]))], dim=0)
                        batch_input = list(batch_input)
                        batch_input[0], batch_input[1], batch_input[2], batch_input[-1] = x_node, x_edge, x_log, y
                        batch_input = tuple(batch_input)

                # forward
                cls_prob, _ = self.model(batch_input, evaluate=True)
                cls_prob = self._fix_prob_shape(cls_prob, B=B, N=N, real_B=real_B)

                # window score/label
                y = y[:real_B]
                y_win = self._window_label(y)
                p_win = self._window_score(cls_prob)

                probs_list.append(p_win.detach().cpu())
                label_list.append(y_win.detach().cpu())

            if not probs_list:
                raise ValueError('Validation loader has no batches')
            probs = torch.cat(probs_list, dim=0)
            labels = torch.cat(label_list, dim=0)

            best = util.search_best_threshold(
                probs, labels,
                steps=self.thr_steps,
                min_thr=self.thr_min,
                max_thr=self.thr_max,
                anom_label=1
            )

            self.threshold = best['thr']
            logging.info(
                f"[THRESHOLD SEARCH] best_thr={best['thr']:.4f} f1={best['f1']:.4f} pr={best['pr']:.4f} rc={best['rc']:.4f}"
            )
            return best

    # --------------------------
    # Imbalance-aware class weight
    # --------------------------
    def _infer_class_weight_from_train_loader(self, train_loader):
        """Compute class weights for CrossEntropyLoss: weight=[w0,w1]."""
        n0, n1 = 0, 0
        for record in train_loader.dataset:
            labels = torch.as_tensor(record['groundtruth_cls'])
            y = labels.argmax(dim=-1).reshape(-1)
            n0 += int((y == 0).sum().item())
            n1 += int((y == 1).sum().item())

        cap = float(self.args.get('imb_wmax', 5.0))
        w0, w1 = class_weights_from_counts(n0, n1, cap)
        class_weight = torch.tensor([w0, w1], device=self.device, dtype=torch.float32)
        logging.info(f"[CLASS WEIGHT] n0={n0} n1={n1} -> weight=[{w0:.3f},{w1:.3f}] (cap={cap})")
        return class_weight

    # --------------------------
    # Training
    # --------------------------
    def fit(self, train_loader, val_loader, **args):
        if len(train_loader) == 0:
            raise ValueError('Training loader has no full batches; reduce batch_size')
        optimizer = AdaBelief(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.learning_change, self.learning_gamma)

        best = {
            "loss": {"score": float("inf"), "state": None, "epoch": 0},
            "f1": {"score": 0, "state": None, "epoch": 0}
        }

        pre_loss, worse_count, isWrong = float("inf"), 0, False

        # classification loss switch
        use_imb = bool(self.args.get('imb_loss', False))
        if use_imb:
            losser = torch.nn.CrossEntropyLoss(
                weight=self._infer_class_weight_from_train_loader(train_loader))
            logging.info('[CLS LOSS] CrossEntropyLoss (imbalance-aware)')
        else:
            losser = torch.nn.CrossEntropyLoss()
            logging.info('[CLS LOSS] CrossEntropyLoss (unweighted ablation)')

        logging.info('optimizer : using AdaBelief')

        for epoch in range(0, self.epoches):
            lr = optimizer.param_groups[0]['lr']
            para = torch.tensor(1 / (epoch // self.rec_down + 1))
            para = para if para > self.para_low else self.para_low

            logging.info('-' * 100)
            logging.info(f'{epoch}/{self.epoches} starting... lr: {lr} para:{para}')

            self.model.train()
            epoch_cls_loss, epoch_rec_loss, epoch_loss = [], [], []
            epoch_time_start = time.time()

            with tqdm(train_loader) as tbar:
                for batch_input in tbar:
                    batch_input = self.input2device(batch_input, self.use_gpu)
                    optimizer.zero_grad()

                    raw_loss, cls_result, cls_label = self.model(batch_input)

                    rec_loss = sum(raw_loss)
                    if cls_result.shape[0] == 0:
                        cls_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
                    else:
                        # cls_label: (M,2) one-hot -> (M,) class id
                        cls_target = cls_label.argmax(dim=1).long() if (cls_label.dim() == 2 and cls_label.size(1) == 2) else cls_label.long()
                        cls_loss = losser(cls_result, cls_target)

                    loss = (1 - para) * cls_loss + para * rec_loss

                    if torch.isnan(loss):
                        isWrong = True
                        logging.info("loss is nan")
                        break

                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
                    optimizer.step()

                    epoch_cls_loss.append(cls_loss.item())
                    epoch_rec_loss.append(rec_loss.item())
                    epoch_loss.append(loss.item())
                    tbar.set_postfix(loss=f'{loss.item():.8f},{cls_loss.item():.8f},{rec_loss.item():.8f}')

            # show epoch results
            epoch_time_elapsed = time.time() - epoch_time_start
            epoch_loss_v = torch.mean(torch.tensor(epoch_loss)).item() if len(epoch_loss) else float('inf')
            epoch_cls_loss_v = torch.mean(torch.tensor(epoch_cls_loss)).item() if len(epoch_cls_loss) else 0.0
            epoch_rec_loss_v = torch.mean(torch.tensor(epoch_rec_loss)).item() if len(epoch_rec_loss) else 0.0

            if isWrong:
                logging.info(f"calculate error in epoch {epoch}")
                break

            if epoch_loss_v <= best["loss"]["score"] or epoch == self.rec_down:
                worse_count = 0
                best["loss"]["score"] = epoch_loss_v
                best["loss"]["state"] = copy.deepcopy(self.model.state_dict())
                best["loss"]["epoch"] = epoch
            elif epoch_loss_v > pre_loss:
                worse_count += 1
                if self.patience > 0 and worse_count >= self.patience:
                    logging.info(f"Early stop at epoch: {epoch}")
                    break

            pre_loss = epoch_loss_v
            logging.info(
                "Epoch {}/{}, all_loss:{:.5f} cls_loss:{:.5f} rec_loss:{:.5f} [{:.2f}s]; best loss:{:.5f}, patience : {}"
                .format(epoch, self.epoches, epoch_loss_v, epoch_cls_loss_v, epoch_rec_loss_v, epoch_time_elapsed, best["loss"]['score'], worse_count)
            )

            # Select checkpoints on validation data; the test split is evaluated
            # only after training and threshold selection in main.py.
            if epoch > self.rec_down:
                result = self.evaluate(val_loader)
                if float(result['f1']) >= best["f1"]["score"]:
                    best["f1"]["score"] = float(result['f1'])
                    best["f1"]["state"] = copy.deepcopy(self.model.state_dict())
                    best["f1"]["epoch"] = epoch

            scheduler.step()

        logging.info('saving model...')
        self.save_model(best['loss'], self.model_save_dir, name='loss')
        self.save_model(best['f1'], self.model_save_dir, name='f1')

    # --------------------------
    # Evaluation (window-level)
    # --------------------------
    def evaluate(self, test_loader, isFinall=False, threshold: float = None):
        self.model.eval()
        with torch.no_grad():
            probs_list, label_list = [], []
            B = self.batch_size

            for batch_input in tqdm(test_loader):
                batch_input = self.input2device(batch_input, self.use_gpu)

                # 1) unpack + preserve structure
                if isinstance(batch_input, dict):
                    x_node = batch_input.get('data_node', None)
                    x_edge = batch_input.get('data_edge', None)
                    x_log = batch_input.get('data_log', None)
                    y = batch_input.get('groundtruth_real', None)
                    if x_node is None or x_edge is None or x_log is None or y is None:
                        raise KeyError(
                            f"batch_input dict keys={list(batch_input.keys())} "
                            f"but expected data_node/data_edge/data_log/groundtruth_real"
                        )
                    extra_dict = {k: v for k, v in batch_input.items()
                                  if k not in ['data_node', 'data_edge', 'data_log', 'groundtruth_real']}
                    original_type = 'dict'
                elif isinstance(batch_input, (list, tuple)):
                    if len(batch_input) < 4:
                        raise ValueError(f"batch_input len={len(batch_input)} < 4, cannot unpack")
                    x_node, x_edge, x_log = batch_input[0], batch_input[1], batch_input[2]
                    y = batch_input[-1]
                    extra_mid = list(batch_input[3:-1])
                    original_type = 'tuple'
                else:
                    raise TypeError(f"Unsupported batch_input type: {type(batch_input)}")

                real_B = x_node.size(0)
                N = x_node.size(2)

                # 2) pad to fixed batch_size
                if real_B < B:
                    pad_B = B - real_B
                    x_node = torch.cat([x_node, x_node.new_zeros((pad_B, *x_node.shape[1:]))], dim=0)
                    x_edge = torch.cat([x_edge, x_edge.new_zeros((pad_B, *x_edge.shape[1:]))], dim=0)
                    x_log = torch.cat([x_log, x_log.new_zeros((pad_B, *x_log.shape[1:]))], dim=0)
                    y = torch.cat([y, y.new_zeros((pad_B, *y.shape[1:]))], dim=0)

                    if original_type == 'dict':
                        extra_dict = {k: torch.cat([v, v.new_zeros((pad_B, *v.shape[1:]))], dim=0)
                                      for k, v in extra_dict.items()}

                # 3) rebuild input
                if original_type == 'dict':
                    model_input = {'data_node': x_node, 'data_edge': x_edge, 'data_log': x_log, 'groundtruth_real': y}
                    model_input.update(extra_dict)
                else:
                    model_input = tuple([x_node, x_edge, x_log] + extra_mid + [y])

                # 4) forward
                raw_result, _ = self.model(model_input, evaluate=True)
                raw_result = self._fix_prob_shape(raw_result, B=B, N=N, real_B=real_B)

                # 5) slice to real batch for labels
                y_real = y[:real_B]

                p_win = self._window_score(raw_result)
                y_win = self._window_label(y_real)

                probs_list.append(p_win.detach().cpu())
                label_list.append(y_win.detach().cpu())

            if not probs_list:
                raise ValueError('Evaluation loader has no batches')
            p_all = torch.cat(probs_list, dim=0).float()
            y_all = torch.cat(label_list, dim=0).long()

            # (M,2) for util.calc_index()
            pred_2col = torch.stack([1.0 - p_all, p_all], dim=1)

            thr = self.threshold if threshold is None else threshold
            info, result = util.calc_index(pred_2col, y_all, threshold=thr)
            return info if isFinall else result
