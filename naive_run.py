import argparse
import os
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from agq.core.scenario import NaiveScenario
from constants import DATA_STREAM_FOLDER, RANKING_FOLDER


def run(x):
    values, ranking_root = x
    scenario = NaiveScenario(values=values)
    scenario.run()
    scenario.summary()
    estimated = scenario.naive.return_top(k=scenario.card)

    pd.DataFrame(estimated).to_csv(
        f'{ranking_root}.csv', index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parameters to run scenario")
    parser.add_argument('--data_folder', type=str, default='./o11ydata/')
    parser.add_argument('--streams_root', type=str, default='')
    parser.add_argument('--multiprocessing', action='store_true')
    args = parser.parse_args()
    print(args)

    # run naive scenario on streams
    streams_filenames = os.listdir(f'{args.data_folder}{DATA_STREAM_FOLDER}/')
    if args.streams_root != '':
        streams_filenames = [e for e in streams_filenames if args.streams_root in streams_filenames]

    if os.path.isdir(f'{args.data_folder}{RANKING_FOLDER}') is False:
        os.makedirs(f'{args.data_folder}{RANKING_FOLDER}')

    if args.multiprocessing is True:
        def process_map_pool(operation, input, max_workers):
            return process_map(operation, input, max_workers=max_workers)


        inputs = []
        for filename in tqdm(streams_filenames):
            values = pd.read_csv(f'{args.data_folder}{DATA_STREAM_FOLDER}/{filename}').values.tolist()

            stream_name = filename.split('.csv')[0]
            ranking_root = f'{args.data_folder}{RANKING_FOLDER}/{stream_name}'
            inputs.append([values, ranking_root])

        out = process_map_pool(run, inputs, os.cpu_count())

    else:
        for filename in tqdm(streams_filenames):
            values = pd.read_csv(f'{args.data_folder}{DATA_STREAM_FOLDER}/{filename}').values.tolist()

            stream_name = filename.split('.csv')[0]
            ranking_root = f'{args.data_folder}{RANKING_FOLDER}/{stream_name}'
            run((values, ranking_root))