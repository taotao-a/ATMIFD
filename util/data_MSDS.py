import logging
import os
import pickle
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from util.constant import *
from util.runtime import CACHE_KEYS, validate_data_paths

# get service dependency
def read_graph(data_dir):
    logging.info("read Graph edge data")
    if not os.path.exists(data_dir):
        logging.info("read no graph data")
        return None
    data = pickle.load(open(os.path.join(data_dir, 'trace_path.pkl'), 'rb'))
    return data

class Process:
    def __init__(self, **kwargs):

        validate_data_paths(kwargs)
        self.cache_config = {key: kwargs[key] for key in CACHE_KEYS}
        self.expected_shapes = {
            'data_node': (kwargs['window'], kwargs['num_nodes'], kwargs['raw_node']),
            'data_edge': (kwargs['window'], kwargs['num_nodes'], kwargs['num_nodes'], kwargs['raw_edge']),
            'data_log': (kwargs['window'], kwargs['num_nodes'], kwargs['log_len']),
            'groundtruth_cls': (kwargs['num_nodes'], 3),
            'groundtruth_real': (kwargs['num_nodes'], 2),
        }

        self.window = kwargs['window']
        self.step = kwargs['step']
        self.dataset_path = kwargs['dataset_path']
        self.rawdata_path = kwargs["data_path"]

        self.log_len = kwargs['log_len']
        self.metric_len = kwargs.get('metric_len', kwargs.get('raw_node', 5))
        self.trace_node_dim = kwargs.get('trace_node_dim', 0)
        self.num_node = kwargs['num_nodes']
        self.percent = kwargs['label_percent']
        self.set, self.dataset, self.trace_type = {}, [], []

        if os.path.exists(self.dataset_path):
            self.read_data()
            self.graph = read_graph(data_dir=self.rawdata_path)
        else:
            self.load_raw()
            self.graph = read_graph(data_dir=self.rawdata_path)
            self.validate_graph()
            logging.info("Tranform data into timewindows")
            self.dataset = self._transform()
            if not self.dataset:
                raise ValueError('Prepared data produced no windows; check the time range and window size')
            for record in self.dataset:
                self.validate_record(record)
            self.save_data()

        self.validate_graph()

    def validate_graph(self):
        if self.graph is None or np.asarray(self.graph).shape != (self.num_node, self.num_node):
            raise ValueError('trace_path.pkl must contain a num_nodes x num_nodes adjacency matrix')
        if not np.isin(self.graph, [0, 1]).all() or not np.asarray(self.graph).any():
            raise ValueError('The service adjacency matrix must be binary and contain at least one edge')

    def validate_record(self, record):
        for key, expected in self.expected_shapes.items():
            actual = np.asarray(record.get(key)).shape
            if actual != expected:
                raise ValueError('Dataset shape mismatch for {}: got {}, expected {}. Select a matching or new --dataset_path.'.format(key, actual, expected))
            
    # reading multidata
    def load_raw(self):
        if not os.path.exists(self.rawdata_path):
            logging.info("Find no data")
        logging.info("LOADing data ...")
        label_raw = pickle.load(open(os.path.join(self.rawdata_path, 'label.pkl'), 'rb'))
        label = np.eye(2)[label_raw.astype(int)]

        label_mask = label_raw.copy()
        times = np.zeros((self.num_node, 2))  
        for idx in range(label.shape[0]):
            if idx < self.window:
                continue
            times += label[idx]
            mask = times[label[idx] == 1] %10 >= 10*self.percent
            label_mask[idx, mask] = 2
        label_mask = np.eye(3)[label_mask.astype(int)]

        metirc = pd.read_csv(os.path.join(self.rawdata_path, 'metric.csv'), sep=',')
        timestart, timeend = metirc['now'].min(), metirc['now'].max()
        time_list = [item for item in range(int(timestart), int(timeend)+1, 1)]
        time_lack = list(set(time_list).difference(set(metirc['now'].values.tolist())))
        for stamp in time_lack:
            metirc = metirc.append([{'now':stamp}])
        metirc = metirc.sort_values(by='now', ascending=True)
        metirc.fillna(method='ffill', inplace=True)
        name_list = list(filter(lambda x: 'mem' in x, list(metirc.columns)))
        after_name = list(map(lambda x: f'{x.split("_")[0]}_a{x.split("_")[-1]}',name_list))
        name_dict = {name_list[idx]: after_name[idx] for idx, _ in enumerate(name_list)}
        metirc.rename(columns = name_dict,  inplace=True)

        # ---- 2-modality mode (trace + metric): disable log modality ----
        # The original implementation loads log.csv and builds a (num_node, log_len)
        # template-count vector per timestamp. For our 2-modality experiments we set
        # log_len=1 (see parser_MSDS.py) and feed all-zero log inputs so the model
        # cannot use log information while keeping code changes minimal.
        log_record = {t: np.zeros((len(MSDS_pod), self.log_len), dtype=np.float32)
                      for t in time_list}

        trace_raw = pd.read_csv(os.path.join(self.rawdata_path, 'trace.csv'), sep=',')
        trace_raw = trace_raw.sort_values(by='end_time', ascending=True)
        self.trace_type.extend(trace_raw['stats'].unique().tolist())
        trace_a = np.zeros((len(MSDS_pod), len(MSDS_pod), len(self.trace_type), len(time_list)))
        for name, item in trace_raw.groupby(['cmbd_id', 'fatherpod', 'stats', 'end_time']):
            if name[0] not in MSDS_pod or name[1] not in MSDS_pod or name[3] > timeend:
                    continue
            trace_a[MSDS_pod.index(name[0]), MSDS_pod.index(name[1]), self.trace_type.index(name[2]), int(name[3]-timestart)] = item['duration'].sum()
        trace = trace_a.transpose(3, 0, 1, 2) / (trace_a.mean(axis=-1)*10 + 1e-6)

        self.set['metric'] = metirc
        self.set['log'] = log_record
        self.set['trace'] = trace
        self.set['label'] = label
        self.set['mask'] = label_mask

    # read data after dealing
    def read_data(self):
        logging.info("read Tranform data")
        if not os.path.exists(self.dataset_path):
            logging.info("read no data")
            return None, None
        
        dataset = [name for name in os.listdir(self.dataset_path)
                   if name.endswith('.pkl') and name[:-4].isdigit()]
        dataset.sort(key=lambda x: int(x[:-4]))
        
        for file in tqdm(dataset):
            data = pickle.load(open(os.path.join(self.dataset_path, file), 'rb'))
            self.validate_record(data)
            self.dataset.append(data)

    # saving data
    def save_data(self):
        logging.info("save Tranform data")
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path, exist_ok=True)
        for _, item in tqdm(enumerate(self.dataset)):
            with open(f'{self.dataset_path}/{item["name"]}.pkl', 'wb') as f:
                del item['name']
                pickle.dump(item, f)
        with open(os.path.join(self.dataset_path, 'cache_config.json'), 'w', encoding='utf-8') as handle:
            json.dump(self.cache_config, handle, indent=2)

    # split to sliding windows
    def _transform(self):
        self.trace_type = list(set(self.trace_type))
        num = 0
        count1, count2, count3 = 0, 0, 0
        data_list = []

        metirc = self.set['metric']
        log = self.set['log']
        trace = self.set['trace']
        label = self.set['label']
        label_mask = self.set['mask']

        starttime, endtime = metirc['now'].min(), metirc['now'].max()

        metirc.sort_index(axis=1, ascending=True, inplace=True)
        
        while starttime + (self.window - 1) * self.step <= endtime:
            record = {}
            if num % 500 == 0:
                logging.info(f"deal ...{num}...trace:{count3}...error see:{count1}...error real:{count2}...{starttime}")
            
            #metric
            select_metirc = metirc[(metirc['now'] >= starttime) & (metirc['now'] <= starttime + (self.window - 1) * self.step)]
            select_metirc.set_index('now', inplace=True)
            select_metirc = select_metirc.values
            select_metirc = select_metirc.reshape(self.window, 5, -1)
            assert select_metirc.shape == (self.window, len(MSDS_pod), 5), f"Worng kpi"
            assert self.metric_len <= select_metirc.shape[-1], \
                f"metric_len({self.metric_len}) > available_metric_dim({select_metirc.shape[-1]})"
            record['data_node'] = select_metirc[:, :, :self.metric_len]

            # log
            log_record = np.stack([log[time] for time in range(int(starttime), int(starttime + (self.window - 1) * self.step + 1), self.step)], axis=0)
            record['data_log'] = np.nan_to_num(log_record)

            assert log_record.shape == (self.window, len(MSDS_pod), self.log_len), f"Worng log"

            #label
            select_label = label[num + self.window - 1, :]
            select_mask = label_mask[num + self.window - 1, :]

            record['groundtruth_cls'] = select_mask
            record['groundtruth_real'] = select_label
            count1 += 1 if record['groundtruth_cls'].sum(axis=0)[1] > 0 else 0
            count2 += 1 if record['groundtruth_real'].sum(axis=0)[1] > 0 else 0
            assert record['groundtruth_cls'].shape == (len(MSDS_pod), 3), f"Worng label"    

            # trace
            select_trace = trace[num : num + self.window]
            count3 += 1 if select_trace.sum() > 0 else 0
            assert select_trace.shape == (self.window, len(MSDS_pod), len(MSDS_pod), len(self.trace_type)), f"Worng Trace"
            # ===== Scheme A: trace -> node structural features (6 dims) + normalization =====
            if getattr(self, "trace_node_dim", 0) == 6:
                # E: (window, N, N) aggregated edge strength (sum over trace types)
                E = select_trace.sum(axis=-1).astype(np.float32)  # (window, N, N)

                eps = 1e-8
                N = E.shape[1]  # number of nodes

                # (window, N)
                out_sum = E.sum(axis=2)  # sum over outgoing edges
                in_sum = E.sum(axis=1)  # sum over incoming edges

                out_deg = (E > 0).sum(axis=2).astype(np.float32)  # outgoing degree count
                in_deg = (E > 0).sum(axis=1).astype(np.float32)  # incoming degree count

                out_max = E.max(axis=2)  # max outgoing strength
                in_max = E.max(axis=1)  # max incoming strength

                out_top1_ratio = out_max / (out_sum + eps)
                in_top1_ratio = in_max / (in_sum + eps)

                # 1) stabilize scale first (log for sums, normalize degree)
                out_sum = np.log1p(out_sum)
                in_sum = np.log1p(in_sum)

                if N > 1:
                    out_deg = out_deg / (N - 1.0)
                    in_deg = in_deg / (N - 1.0)

                trace_node_feat = np.stack([
                    out_sum,
                    in_sum,
                    out_deg,
                    in_deg,
                    out_top1_ratio,
                    in_top1_ratio
                ], axis=-1).astype(np.float32)  # (window, N, 6)

                # 2) z-score normalization per window (over nodes), feature-wise
                # mean/std: (window, 1, 6)
                mu = trace_node_feat.mean(axis=1, keepdims=True)
                sigma = trace_node_feat.std(axis=1, keepdims=True)
                trace_node_feat = (trace_node_feat - mu) / (sigma + 1e-6)

                # concat to metric node features: (window, N, metric_len + 6)
                record['data_node'] = np.concatenate([record['data_node'], trace_node_feat], axis=-1)

            record['data_edge'] = select_trace
            record['name'] = f'{num}'
            num += 1
            data_list.append(record)
            starttime += self.step
            del record
        logging.info(f"deal ...{num}...error see:{count1}...error real:{count2}...")
        return data_list
