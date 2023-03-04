import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from pathlib import Path

from agq.core.scenario import NaiveScenario


def uniform(card, scale=100):
    counts = 1 + np.random.exponential(scale=scale, size=card).astype(int)
    lefts = np.maximum(np.random.normal(loc=0.0, scale=1.0, size=card), 0)
    rights = np.maximum(np.random.normal(loc=2.0, scale=1.0, size=card), 0)

    values = []
    for i in range(card):
        key = 'key_{}'.format(i)
        count, left, right = counts[i], lefts[i], max(rights[i], lefts[i] + 0.5)
        for value in np.random.uniform(low=left, high=right, size=count):
            values.append([key, value])
    assert card == len(set([e[0] for e in values]))

    print(f'''Tag has {args.card} values and {len(values)} records, on average {len(values) / args.card : 2f}''')

    random.shuffle(values)
    return values


def generate(card, distribution, scale=100):
    if distribution == 'UNIFORM':
        return uniform(card, scale=scale)
    else:
        raise NotImplementedError


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parameters to generate synthetic data")
    parser.add_argument('--n_streams', type=int, default=10)
    parser.add_argument('--card', type=int, default=1000)
    parser.add_argument('--distribution', type=str, default='UNIFORM')
    parser.add_argument('--scale', type=float, default=100)
    parser.add_argument('--p', type=float, default=90)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed + 100)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    for i in tqdm(range(args.n_streams)):
        values = generate(args.card, args.distribution, scale=args.scale)

        naive_scenario = NaiveScenario(values=values)
        naive_scenario.run()

        ranking = naive_scenario.naive.return_top(p=args.p, k=args.card)

        streams_path = Path(__file__).parent / 'streams'
        rankings_path = Path(__file__).parent / 'rankings'
        if streams_path.exists() is False:
            streams_path.mkdir()
        if rankings_path.exists() is False:
            rankings_path.mkdir()

        pd.DataFrame(values, columns=["idx", "value"]).to_csv(streams_path / f'synthetic_card_{args.card}_{timestamp}_{i}.csv', index=False)
        pd.DataFrame(ranking).to_csv(rankings_path / f'synthetic_card_{args.card}_{timestamp}_{i}.csv', index=False)







