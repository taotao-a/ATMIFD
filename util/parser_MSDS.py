import argparse


def str_bool(value):
    if value.lower() not in ('true', 'false'):
        raise argparse.ArgumentTypeError('expected true or false')
    return value.lower() == 'true'

parser = argparse.ArgumentParser(description='ATMIFD: metric/trace anomaly detection', allow_abbrev=False)
parser.add_argument("--random_seed", default=42,
                    type=int, help='the random seed')

# training setting
parser.add_argument("--gpu", default=True, type=str_bool,
                    help='legacy device flag; --device takes precedence')
parser.add_argument("--device", default='auto', choices=['auto', 'cpu', 'cuda'],
                    help='auto uses CUDA when available, otherwise CPU')
parser.add_argument("--epochs", default=50, type=int,
                    help='the number of training epochs')
parser.add_argument("--patience", default=15, type=float,
                    help='the number of epoch that loss is uping')
parser.add_argument("--learning_rate", default=1e-3,
                    type=float, help='the data number at one epoch')
parser.add_argument("--weight_decay", default=5e-4, type=float,
                    help='the one of optimzier parameters which prevent overfitting ')
parser.add_argument("--learning_change", default=100, type=int,
                    help='the epoch number that change learning rate')
parser.add_argument("--learning_gamma", default=0.9, type=float,
                    help='the weight that change learning rate')
parser.add_argument("--label_weight", default=1e-2, type=float,
                    help='the unkown weight in reconstruction loss')
parser.add_argument("--label_percent", default=0.5, type=float,
                    help='the proportion of labeled data') 
parser.add_argument("--rec_down", default=1, type=int,
                    help='the number that changes reconstruction loss weight')
parser.add_argument("--para_low", default=1e-2, type=float,
                    help='the min weight of rec loss')

# model setting
parser.add_argument("--feature_node", default=4, type=int,
                    help='the pod kpi data number at one epoch')
parser.add_argument("--feature_edge", default=4, type=int,
                    help='the edge data number at one epoch')
parser.add_argument("--feature_log", default=16, type=int,
                    help='the log data number at one epoch')
parser.add_argument("--metric_len", default=5, type=int,
                    help="number of metric features (pure metric dim)")
parser.add_argument("--trace_node_dim", default=6, type=int,
                    help="extra trace-derived node feature dim for scheme A")

parser.add_argument("--raw_node", default=None, type=int,
                    help="total node feature dim fed into model; if None, metric_len + trace_node_dim")

parser.add_argument("--raw_edge", default=7, type=int,
                    help='the raw edge kpi data number at one epoch')
# NOTE: for 2-modality (trace + metric) experiments, we disable the log modality
# by shrinking log feature length to 1 and feeding all-zero log inputs in data_MSDS.py.
parser.add_argument("--log_len", default=1, type=int,
                    help='the log template amount (set to 1 to disable logs)')

parser.add_argument("--num_heads_edge", default=4, type=int,
                    help='the number of multiattention heads about trace')
parser.add_argument("--num_heads_node", default=4, type=int,
                    help='the number of multiattention heads about metric')
parser.add_argument("--num_heads_log", default=4, type=int,
                    help='the number of multiattention heads about log')
parser.add_argument("--num_heads_n2e", default=4, type=int,
                    help='the number of multiattention heads about node')
parser.add_argument("--num_heads_e2n", default=2, type=int,
                    help='the number of multiattention heads about edge')
parser.add_argument("--num_layer", default=2, type=int,
                    help='the number of model layers')
parser.add_argument("--dropout", default=0.2, type=float)


# dataset setting
parser.add_argument("--batch_size", default=50, type=int,
                    help='the data number at one epoch')
parser.add_argument("--window", default=10, type=int,
                    help='size of sliding window')
parser.add_argument("--step", default=1, type=int,
                    help='sliding window stride')
parser.add_argument("--num_nodes", default=5, type=int,
                    help='the number of node in graph')

# path setting
parser.add_argument("--data_path", default='./data/MSDS-pre',
                    type=str, help='the path of raw data')
parser.add_argument("--dataset_path", default=None,
                    type=str, help='the path of saving data')
parser.add_argument("--result_dir", default="./result",
                    type=str, help='the path of result and log')

parser.add_argument("--main_model", default='ATMIFD', type=str,
                    help='experiment name used for the output directory')
parser.add_argument("--evaluate", default=False, 
                    type=str_bool, help='Evaluate the exist model')
parser.add_argument("--model_path", default=None,
                    type=str, help=' the path of exist model')


# threshold setting (decision on anomaly probability)
parser.add_argument("--threshold", default=None, type=float,
                    help="anomaly decision threshold on P(class=1); if None, use argmax (equivalent to 0.5 for 2-class probs)")
parser.add_argument("--thr_search", default=True, type=str_bool,
                    help="whether to search best threshold on validation set")
parser.add_argument("--thr_min", default=0.0, type=float, help="min threshold for grid search")
parser.add_argument("--thr_max", default=1.0, type=float, help="max threshold for grid search")
parser.add_argument("--thr_steps", default=200, type=int, help="number of grid points for threshold search")

# ===== Ablation switches =====
parser.add_argument("--imb_loss", default=True, type=str_bool,
                    help="use imbalance-aware classification loss (CrossEntropy + class weight)")
parser.add_argument("--imb_wmax", default=5.0, type=float,
                    help="cap anomaly weight w1=min(normal/anomaly, cap)")

# window-level aggregation for anomaly score p(window)
parser.add_argument("--win_agg", type=str, default="max", choices=["max", "topk", "lse"],
                    help="aggregation over nodes to get window-level anomaly score")
parser.add_argument("--topk", type=int, default=3,
                    help="k for top-k mean aggregation when win_agg=topk")
parser.add_argument("--lse_tau", type=float, default=0.2,
                    help="temperature for log-sum-exp aggregation when win_agg=lse")

# evaluation output
parser.add_argument("--eval_stage", type=str, default="f1", choices=["loss", "f1", "both"],
                    help="which checkpoint to evaluate after training")

def parse_args(argv=None):
    return vars(parser.parse_args(argv))


def validate_args(args):
    for key in ('epochs', 'batch_size', 'window', 'rec_down', 'learning_change',
                'num_layer', 'metric_len', 'raw_edge', 'log_len', 'topk',
                'feature_node', 'feature_edge', 'feature_log', 'num_heads_node',
                'num_heads_edge', 'num_heads_log', 'num_heads_n2e', 'num_heads_e2n'):
        if args[key] <= 0:
            raise ValueError('--{} must be positive'.format(key))
    if args['step'] != 1 or args['num_nodes'] != 5:
        raise ValueError('This MSDS implementation supports --step 1 and --num_nodes 5 only')
    if args['trace_node_dim'] not in (0, 6) or not 1 <= args['metric_len'] <= 5:
        raise ValueError('--trace_node_dim must be 0 or 6; --metric_len must be between 1 and 5')
    expected = args['metric_len'] + args['trace_node_dim']
    if args['raw_node'] is None:
        args['raw_node'] = expected
    if args['raw_node'] != expected:
        raise ValueError('--raw_node must equal metric_len + trace_node_dim')
    for feature, heads in (('feature_node', 'num_heads_node'),
                           ('feature_edge', 'num_heads_edge'),
                           ('feature_log', 'num_heads_log'),
                           ('feature_edge', 'num_heads_e2n')):
        if args[feature] % 2 or args[feature] % args[heads]:
            raise ValueError('{} must be even and divisible by {}'.format(feature, heads))
    if (args['feature_node'] + args['feature_log']) % args['num_heads_n2e']:
        raise ValueError('feature_node + feature_log must be divisible by num_heads_n2e')
    if not 0 <= args['label_percent'] <= 1 or not 0 <= args['dropout'] < 1:
        raise ValueError('label_percent must be in [0,1]; dropout must be in [0,1)')
    if not 0 <= args['para_low'] <= 1 or args['label_weight'] < 0:
        raise ValueError('para_low must be in [0,1]; label_weight must be nonnegative')
    if args['learning_rate'] <= 0 or args['weight_decay'] < 0 or args['learning_gamma'] <= 0:
        raise ValueError('learning_rate and learning_gamma must be positive; weight_decay must be nonnegative')
    if args['imb_wmax'] < 1:
        raise ValueError('invalid class-weight parameters')
    if args['thr_steps'] < 2 or not 0 <= args['thr_min'] < args['thr_max'] <= 1:
        raise ValueError('threshold search needs at least 2 points and 0 <= thr_min < thr_max <= 1')
    if args['threshold'] is not None and not 0 <= args['threshold'] <= 1:
        raise ValueError('--threshold must be in [0,1]')
    if args['lse_tau'] <= 0 or args['window'] > 1000:
        raise ValueError('lse_tau must be positive and window must not exceed 1000')
    if not args['evaluate'] and args['eval_stage'] in ('f1', 'both') and args['epochs'] <= args['rec_down'] + 1:
        raise ValueError('F1 checkpoint selection requires epochs > rec_down + 1; use --eval_stage loss for a shorter run')
    if not args['main_model'] or any(c in args['main_model'] for c in '/\\:*?"<>|'):
        raise ValueError('--main_model must be a valid directory name')
