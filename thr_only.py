"""Search a validation-set threshold for a trusted saved experiment."""

import logging

from util.parser_MSDS import parse_args
from util.runtime import prepare_args, build_loaders


def main(argv=None):
    args = prepare_args(parse_args(argv), argv, threshold_only=True)
    import util.util as util
    from util.train import MY
    from util.data_MSDS import Process
    from src.model import MyModel

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s', force=True)
    util.seed_everything(args['random_seed'])
    processed = Process(**args)
    _, val_dl, _ = build_loaders(processed, args)
    trainer = MY(MyModel(processed.graph, **args), **args)
    trainer.load_model(args['model_path'], name=args['eval_stage'])
    best = trainer.search_threshold(val_dl)
    print('Validation threshold: {thr:.6f}; F1: {f1:.4f}; precision: {pr:.4f}; recall: {rc:.4f}'.format(**best))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as error:
        raise SystemExit('ATMIFD: {}'.format(error))
