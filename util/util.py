import hashlib
import json
import logging
import os
import time
import pickle
import random
import numpy as np
import torch
from sklearn.metrics import *
from util.constant import *

import numpy as np
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

def _to_numpy(x):
    """Convert list/tensor/ndarray to numpy array safely."""
    if isinstance(x, list):
        if len(x) == 0:
            return np.array([])
        if torch.is_tensor(x[0]):
            x = torch.cat([t.detach().cpu() for t in x], dim=0)
            return x.numpy()
        else:
            return np.asarray(x)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def search_best_threshold(score1, y_true, steps: int = 200, min_thr: float = 0.0, max_thr: float = 1.0, anom_label: int = 1):
    """Grid-search threshold on validation set to maximize F1 for anomaly label (default=1).
    score1: anomaly probability (N,)
    y_true: ground-truth labels (N,)
    """
    score1 = _to_numpy(score1).reshape(-1)
    y_true = _to_numpy(y_true).reshape(-1).astype(np.int64)

    best = {"thr": 0.5, "f1": -1.0, "pr": 0.0, "rc": 0.0}
    if score1.size == 0 or y_true.size == 0:
        return best

    for thr in np.linspace(min_thr, max_thr, steps):
        y_pred = (score1 >= thr).astype(np.int64)
        pr = precision_score(y_true, y_pred, pos_label=anom_label, zero_division=0)
        rc = recall_score(y_true, y_pred, pos_label=anom_label, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=anom_label, zero_division=0)
        if f1 > best["f1"]:
            best = {"thr": float(thr), "f1": float(f1), "pr": float(pr), "rc": float(rc)}
    return best

def calc_index(predict, actual, threshold: float = None):
    """
    predict: (M,2) probs/logits OR list of (B,2) tensors
    actual : (M,2) one-hot OR (M,) labels OR list
    Return: (info_str, metrics_dict) to keep compatibility with your trainer.
    """
    pred_np = _to_numpy(predict)
    act_np  = _to_numpy(actual)

    if act_np.ndim > 2:
        act_np = act_np.reshape(-1, act_np.shape[-1])
    if pred_np.ndim > 2:
        pred_np = pred_np.reshape(-1, pred_np.shape[-1])

    # y_true
    if act_np.ndim == 2 and act_np.shape[-1] == 2:
        y_true = np.argmax(act_np, axis=-1)
    else:
        y_true = act_np.reshape(-1)

    # y_pred + scores
    score1 = None
    if pred_np.ndim == 2 and pred_np.shape[-1] == 2:
        score1 = pred_np[:, 1]
        if threshold is None:
            y_pred = np.argmax(pred_np, axis=-1)
        else:
            # threshold over anomaly prob (class=1)
            y_pred = (score1 >= float(threshold)).astype(np.int64)
    else:
        y_pred = pred_np.reshape(-1)

    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    def pr_rc_f1_for(anom_label: int):
        pr = precision_score(y_true, y_pred, pos_label=anom_label, zero_division=0)
        rc = recall_score(y_true, y_pred, pos_label=anom_label, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=anom_label, zero_division=0)
        return pr, rc, f1

    pr1, rc1, f11 = pr_rc_f1_for(1)
    pr0, rc0, f10 = pr_rc_f1_for(0)

    # Anomaly is always class 1. Do not switch to the majority/normal class:
    # average precision is class-dependent even when binary AUCs are equal.
    auc = ap = 0.0
    if score1 is not None:
        anomaly_target = (y_true == 1).astype(int)
        try:
            auc = float(roc_auc_score(anomaly_target, score1))
        except Exception:
            auc = 0.0
        try:
            ap = float(average_precision_score(anomaly_target, score1))
        except Exception:
            ap = 0.0

    total = len(y_true)
    pred_wrong = int((y_pred != y_true).sum())
    pred_right = int(total - pred_wrong)
    actu_right = int((y_true == 0).sum())
    actu_wrong = int((y_true == 1).sum())

    info = (
        f"anom=1 pr:{pr1:.4f} rc:{rc1:.4f} f1:{f11:.4f} | "
        f"anom=0 pr:{pr0:.4f} rc:{rc0:.4f} f1:{f10:.4f} | "
        f"auc:{auc:.4f} ap:{ap:.4f} "
        f"pred_right:{pred_right} pred_wrong:{pred_wrong} "
        f"actu_right:{actu_right} actu_wrong:{actu_wrong}"
    )
    logging.info(info)

    main_pr, main_rc, main_f1 = pr1, rc1, f11

    return info, {
        "pr": float(main_pr),
        "rc": float(main_rc),
        "f1": float(main_f1),
        "auc": float(auc),
        "ap": float(ap),
        "pr_anom_is_1": float(pr1), "rc_anom_is_1": float(rc1), "f1_anom_is_1": float(f11),
        "pr_anom_is_0": float(pr0), "rc_anom_is_0": float(rc0), "f1_anom_is_0": float(f10),
    }

def json_pretty_dump(obj, filename):
    with open(filename, "w", encoding="utf-8") as fw:
        json.dump(obj, fw, sort_keys=True, indent=4,
                  separators=(",", ": "), ensure_ascii=False, )


def dump_params(args):
    hash_id = hashlib.md5(str(sorted([(k, v) for k, v in args.items()])).encode("utf-8")).hexdigest()[0:8]
    save_path = os.path.join(args['result_dir'], args['main_model'] + '-' + os.path.basename(os.path.normpath(args['dataset_path'])) + '-' + hash_id + '-' + str(int(time.time())))
    os.makedirs(save_path, exist_ok=True)

    log_file = os.path.join(save_path, "running.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()

    logging.basicConfig(
        level=logging.INFO,  
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return hash_id, save_path


def read_params(args):
    filename = os.path.join(args['model_path'], "params.json")
    with open(filename, encoding="utf-8") as f:
        dict_json = json.load(fp=f)
   
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )

    return dict_json


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dump_pickle(obj, file_path):
    logging.info("Dumping to {}".format(file_path))
    with open(file_path, "wb") as fw:
        pickle.dump(obj, fw)


def load_pickle(file_path):
    logging.info("Loading from {}".format(file_path))
    with open(file_path, "rb") as fr:
        return pickle.load(fr)
