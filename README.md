# Approximate Top Grouped Quantiles on Streams

This repository contains reference Python implementations for the "high rollers" data structures described in our research article, as well as scripts for reproducing the results on synthetic datasets.

## Setup
From `approximate-grouped-quantiles` directory:

```
python -m venv <your_venv_name>
source <your_venv_name>/bin/activate
pip install -e .
```

To run the notebooks:
```
pip install ipykernel
python -m ipykernel install --user --name=<your_venv_name>
cd notebooks 
jupyter lab
```

To run the unit tests locally:
```
pytest tests -rx
```


## Synthetics

Examples:
```
python synthetics/generate.py --n_streams 12 --card 1000 --distribution UNIFORM --p 90
python scenario_run.py --data_folder ./synthetics/ --scenario_name '' --streams_root '' --num_digests 200 --prob_method UPDATING --cm --n_runs 0 --multiprocessing
python scenario_eval.py --path_to_scenario ./synthetics/scenarios/scenario_0
```

For convenience, the synthetic datasets and results of the research article can be reproduced via simple `make` commands: `make gen_data`, `make run_many`, and `make run_merge`. These can take a while, so feel free to run with smaller parameters to iron out any kinks. In any case these will produce (many) data files in `synthetics/streams`, the true rankings in `synthetics/rankings`, and the results of the scenarios (including their descriptions) in `synthetics/scenarios`.

To obtain the tables from the paper, you will need to run the [notebook](notebooks/scores_from_paper.ipynb), specifying the "path to scenario" you want to include for analysis (e.g., '../synthetics/scenarios/scenario_5', etc.) as well as the "stream root" (e.g., 'card_10000_202303...')


## License
See the [license](LICENSE).
