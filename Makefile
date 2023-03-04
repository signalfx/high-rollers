gen_data:
	python synthetics/generate.py --card 1000 --distribution UNIFORM --p 90
	python synthetics/generate.py --card 5000 --distribution UNIFORM --p 90
	python synthetics/generate.py --card 10000 --distribution UNIFORM --p 90

run_many:
	python scenario_run.py --data_folder ./synthetics/ --scenario_name '' --streams_root '' --num_digests 200 --ss --ss_capacity 500 --prob_method UPDATING --n_runs 10 --multiprocessing --seed 1
	python scenario_run.py --data_folder ./synthetics/ --scenario_name '' --streams_root '' --num_digests 200 --ss --ss_capacity 500 --prob_method UPDATING --truncate_churn --n_runs 10 --multiprocessing --seed 1
	python scenario_run.py --data_folder ./synthetics/ --scenario_name '' --streams_root '' --num_digests 200 --ss --ss_capacity 500 --prob_method UNIFORM --n_runs 10 --multiprocessing --seed 1
	python scenario_run.py --data_folder ./synthetics/ --scenario_name '' --streams_root '' --num_digests 200 --ss --ss_capacity 500  --prob_method UNIFORM --truncate_churn --n_runs 10 --multiprocessing --seed 1
	python scenario_run.py --data_folder ./synthetics/ --scenario_name '' --streams_root '' --num_digests 200 --dummy --n_runs 10 --multiprocessing --seed 1
	python scenario_run.py --data_folder ./synthetics/ --scenario_name '' --streams_root '' --num_digests 200 --n_runs 10 --multiprocessing --seed 1
	python scenario_run.py --data_folder ./synthetics/ --scenario_name '' --streams_root '' --num_digests 200 --n_runs 10 --multiprocessing --cm --cm_width 200 --cm_depth 10 --seed 1


run_merge:
	python synthetics/generate.py --card 1000 --distribution UNIFORM --p 90 --scale 500 --n_streams 5
	# you might filter down to the streams just generated via "--streams_root <card_1000_YYYYMM....>", else this will run on many datasets
	python scenario_run.py --data_folder ./synthetics/ --scenario_name '' --streams_root '' --num_digests 200 --n_runs 10 --ss --ss_capacity 500 --seed 1 --merge_opn
