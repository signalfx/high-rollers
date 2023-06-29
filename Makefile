gen_data:
	python synthetics/generate.py --card 1000 --distribution UNIFORM --p 90
	python synthetics/generate.py --card 10000 --distribution UNIFORM --p 90

run_many:
	python scenario_run.py --data_folder ./synthetics/ --scenario_name '' --streams_root '' --num_digests 200 --ss --ss_capacity 500 --prob_method UPDATING --n_runs 1 --multiprocessing --seed 1
	python scenario_run.py --data_folder ./synthetics/ --scenario_name '' --streams_root '' --num_digests 200 --ss --ss_capacity 500 --prob_method UPDATING --truncate_churn --n_runs 1 --multiprocessing --seed 1
	python scenario_run.py --data_folder ./synthetics/ --scenario_name '' --streams_root '' --num_digests 200 --ss --ss_capacity 500 --prob_method UNIFORM --n_runs 1 --multiprocessing --seed 1
	python scenario_run.py --data_folder ./synthetics/ --scenario_name '' --streams_root '' --num_digests 200 --ss --ss_capacity 500  --prob_method UNIFORM --truncate_churn --n_runs 1 --multiprocessing --seed 1
	python scenario_run.py --data_folder ./synthetics/ --scenario_name '' --streams_root '' --num_digests 200 --dummy --n_runs 1 --multiprocessing --seed 1
	python scenario_run.py --data_folder ./synthetics/ --scenario_name '' --streams_root '' --num_digests 200 --n_runs 1 --multiprocessing --seed 1
	python scenario_run.py --data_folder ./synthetics/ --scenario_name '' --streams_root '' --num_digests 200 --n_runs 1 --multiprocessing --cm --cm_width 200 --cm_depth 10 --seed 1

run_merge:
	python synthetics/generate.py --card 1000 --distribution UNIFORM --p 90 --scale 1000 --n_streams 10
	# you might filter down to the streams just generated via "--streams_root <card_1000_YYYYMM....>", else this will run on many datasets
	python scenario_run.py --data_folder ./synthetics/ --scenario_name '' --streams_root '' --num_digests 200 --n_runs 1 --ss --ss_capacity 500 --seed 1 --merge_opn
