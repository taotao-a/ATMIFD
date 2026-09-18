"""Train ATMIFD or evaluate a saved experiment."""

import logging
import os

from util.parser_MSDS import parse_args
from util.runtime import prepare_args, build_loaders


def main(argv=None):
    args = prepare_args(parse_args(argv), argv)
    # Lazy imports allow --help and input checks without ML dependencies.
    import util.util as util
    import util.train as train
    from util.data_MSDS import Process
    from src.model import MyModel

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s', force=True)
    util.seed_everything(args['random_seed'])
    processed = Process(**args)
    train_dl, val_dl, test_dl = build_loaders(processed, args)
    if not args['evaluate']:
        args['hash_id'], args['result_dir'] = util.dump_params(args)
        args['model_path'] = args['result_dir']
        util.json_pretty_dump(args, os.path.join(args['result_dir'], 'params.json'))

    sample = processed.dataset[0]
    logging.info('[DATA] windows=%s node=%s edge=%s log=%s', len(processed.dataset),
                 sample['data_node'].shape, sample['data_edge'].shape, sample['data_log'].shape)
    trainer = train.MY(MyModel(processed.graph, **args), **args)
    if not args['evaluate']:
        trainer.fit(train_loader=train_dl, val_loader=val_dl)

    stages = ['loss', 'f1'] if args['eval_stage'] == 'both' else [args['eval_stage']]
    with open(os.path.join(args['result_dir'], 'evaluation.log'), 'a', encoding='utf-8') as handle:
        for stage in stages:
            trainer.load_model(args['model_path'], name=stage)
            if args['thr_search']:
                trainer.search_threshold(val_dl)
            info = trainer.evaluate(test_dl, isFinall=True)
            handle.write('{} {} | thr={}\n'.format(stage, info, trainer.threshold))
    logging.info('Results saved in %s', args['result_dir'])


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as error:
        raise SystemExit('ATMIFD: {}'.format(error))
