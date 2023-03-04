import argparse
from collections import defaultdict
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from agq.metrics import get_rank_error, get_hit_ratio
from constants import DATA_STREAM_FOLDER, RANKING_FOLDER, EXCLUDED_FILENAMES

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parameters to evaluate scenario")
    parser.add_argument('--path_to_scenario', type=str, default='./o11ydata/scenarios/scenario_0')
    parser.add_argument('--streams_root', type=str, default='')
    parser.add_argument('--k', type=int, default=-1)
    args = parser.parse_args()
    print(args)

    # load params
    with open(f'{args.path_to_scenario}/params.json', 'r') as f:
        params = json.load(f)
        params.update(vars(args))
        args = argparse.Namespace(**params)

    streams_filenames = sorted(os.listdir(f'{args.data_folder}{DATA_STREAM_FOLDER}/'))
    if args.streams_root != '':
        streams_filenames = [e for e in streams_filenames if args.streams_root in e]

    dataframe = []
    index = []
    list_topK = [200, 175, 150, 125, 100, 75, 50, 25]
    for filename in tqdm(streams_filenames):
        if filename in EXCLUDED_FILENAMES:
            continue
        scores = defaultdict(list)
        for topK in list_topK:
            stream_name = filename.split('.csv')[0]
            truth_ranking = pd.read_csv(f'{args.data_folder}{RANKING_FOLDER}/{filename}')

            filename_scores, counter = defaultdict(list), 0
            while os.path.isfile(f'{args.path_to_scenario}/{stream_name}_ranking_{counter}.csv'):
                estimated_ranking = pd.read_csv(
                    f'{args.path_to_scenario}/{stream_name}_ranking_{counter}.csv')

                if topK != -1:
                    estimated_ranking = estimated_ranking.iloc[:topK-1]

                filename_scores['rank_error'].append(
                    get_rank_error(truth_ranking=truth_ranking, estimated_ranking=estimated_ranking, weighted=False)
                )
                # filename_scores['hit_ratio'].append(
                #     get_hit_ratio(truth_ranking=truth_ranking, estimated_ranking=estimated_ranking)
                # )
                counter += 1

            for k, v in filename_scores.items():
                v_array = np.array(v)
                scores[k].append((v_array.mean(), v_array.std()))

        if len(scores) > 0:
            index.append(filename)
            row = []
            for mean, std in scores['rank_error']:
                row.append(mean)
                row.append(std)
            dataframe.append(row)

    columns = []
    for topK in list_topK:
        columns.append("rank error top {} (mean)".format(topK))
        columns.append("rank error top {} (std)".format(topK))

    df_scores = pd.DataFrame(dataframe, columns=columns, index=index)
    df_scores.to_csv(f'{args.path_to_scenario}/scores.csv')
    print(df_scores)
