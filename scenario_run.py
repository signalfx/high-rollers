import argparse
import json
import os
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from agq.core.scenario import Scenario
from constants import DATA_STREAM_FOLDER, SCENARIO_FOLDER


def run(x, merge_opn=False):
    values, num_digests, prob_method, truncate_churn, cm, width, depth, ss, ss_capacity, dummy, ranking_root = x

    keys = list(set([e[0] for e in values]))
    key2id = {key: i for i, key in enumerate(keys)}
    id2key = {i: key for i, key in enumerate(keys)}
    new_values = [[key2id[e[0]], e[1]] for e in values]

    if merge_opn is False:
        s_pa = Scenario(values=new_values, num_digests=num_digests, prob_method=prob_method,
                        truncate_churn=truncate_churn, cm=cm, width=width, depth=depth, ss=ss,
                        ss_capacity=ss_capacity)
        s_pa.run(cm=cm, dummy=dummy, ss=ss)



        s_pa.summary()
        estimated = s_pa.agq.return_top(k=s_pa.agq.max_digests)
    else:
        num_agqs = 10
        list_agq = []
        random.shuffle(new_values)
        for i in range(num_agqs):
            left, right = i * (len(new_values) // num_agqs),  (i+1) * (len(new_values) // num_agqs)
            s_pa = Scenario(values=new_values[left:right], num_digests=num_digests, prob_method=prob_method,
                            truncate_churn=truncate_churn, cm=cm, width=width, depth=depth, ss=ss,
                            ss_capacity=ss_capacity)
            s_pa.run(cm=cm, dummy=dummy, ss=ss)
            list_agq.append(s_pa.agq)

        agq = list_agq[0]
        for other_agq in list_agq[1:]:
            agq.merge(other_agq)

        estimated = agq.return_top(k=agq.max_digests)

    counter = 0
    while os.path.isfile(f'{ranking_root}{counter}.csv'):
        counter += 1

    estimated_df = pd.DataFrame(estimated)
    estimated_df['idx'] = estimated_df.apply(lambda x: id2key[int(x['idx'])], axis=1)
    estimated_df.to_csv(f'{ranking_root}{counter}.csv', index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parameters to run scenario")
    parser.add_argument('--data_folder', type=str, default='./synthetics/')
    parser.add_argument('--scenario_name', type=str, default='')
    parser.add_argument('--streams_root', type=str, default='')
    parser.add_argument('--num_digests', type=int, default=200)

    parser.add_argument('--prob_method', type=str, default='UPDATING')
    parser.add_argument('--truncate_churn', action='store_true')

    parser.add_argument('--cm', action='store_true')
    parser.add_argument('--cm_width', type=int, default=1000)
    parser.add_argument('--cm_depth', type=int, default=20)

    parser.add_argument('--ss', action='store_true')
    parser.add_argument('--ss_capacity', type=int, default=1000)

    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true')
    parser.add_argument('--dummy', action='store_true')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--merge_opn', action='store_true')
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)

    if os.path.isdir(f'{args.data_folder}{SCENARIO_FOLDER}') is False:
        os.makedirs(f'{args.data_folder}{SCENARIO_FOLDER}')

    # if scenario name is not given
    if args.scenario_name == '':
        scenarios = os.listdir(f'{args.data_folder}{SCENARIO_FOLDER}/')

        if len(scenarios) == 0:
            counter = 0
        else:
            counter = 1 + max([int(e.split('_')[1]) for e in scenarios])
        args.scenario_name = f'scenario_{str(counter)}'
        os.makedirs(f'{args.data_folder}{SCENARIO_FOLDER}/{args.scenario_name}/')

        with open(f'{args.data_folder}{SCENARIO_FOLDER}/{args.scenario_name}/params.json', 'w') as outfile:
            outfile.write(json.dumps(vars(args), indent=4))

    # else load scenario parameters
    else:
        with open(f'{args.data_folder}{SCENARIO_FOLDER}/{args.scenario_name}/params.json', 'r') as f:
            params = json.load(f)
            args = argparse.Namespace(**params)

    # run scenario on streams
    streams_filenames = sorted(os.listdir(f'{args.data_folder}{DATA_STREAM_FOLDER}/'))
    if args.streams_root != '':
        streams_filenames = [e for e in streams_filenames if args.streams_root in e]

    if args.multiprocessing is True:
        if args.merge_opn is True:
            raise NotImplementedError
        def process_map_pool(operation, input, max_workers):
            return process_map(operation, input, max_workers=max_workers)


        inputs = []
        for filename in tqdm(streams_filenames):
            values = pd.read_csv(f'{args.data_folder}{DATA_STREAM_FOLDER}/{filename}').values.tolist()
            stream_name = filename.split('.csv')[0]

            for _ in range(args.n_runs):
                if args.shuffle:
                    random.shuffle(values)

                ranking_root = f'{args.data_folder}{SCENARIO_FOLDER}/{args.scenario_name}/{stream_name}_ranking_'
                inputs.append(
                    [values, args.num_digests, args.prob_method, args.truncate_churn, args.cm,
                     args.cm_width, args.cm_depth, args.ss, args.ss_capacity,
                     args.dummy, ranking_root])

        out = process_map_pool(run, inputs, os.cpu_count())

    else:
        for filename in tqdm(streams_filenames):
            values = pd.read_csv(f'{args.data_folder}{DATA_STREAM_FOLDER}/{filename}').values.tolist()

            for _ in range(args.n_runs):
                if args.shuffle:
                    random.shuffle(values)

                stream_name = filename.split('.csv')[0]
                ranking_root = f'{args.data_folder}{SCENARIO_FOLDER}/{args.scenario_name}/{stream_name}_ranking_'
                run((values, args.num_digests, args.prob_method, args.truncate_churn, args.cm,
                     args.cm_width, args.cm_depth, args.ss, args.ss_capacity, args.dummy,
                     ranking_root), merge_opn=args.merge_opn)
