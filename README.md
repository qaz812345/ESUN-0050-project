# ESUN 0050 Project

## Introduction
Learning portfolio management with a deep reinforcement learning based method.
 

## Dateset Generation
```
python data_prepocess.py --dataset s10 --start 1995-01-01 --end 2004-12-31 --output data_file_name
```

## Train DMP
* Create a csv with the same format as the result_template.csv and name it with the same name as the file_name args you set in train_dmp.py.

* start an experiment with 
```
python train_dmp.py --num_steps 500 --n_episode 128 --episode_step 50 --lr 3e-6 --win_size 31 --reg_w 1e-4 --rolling --num_rolling_steps 10 --file_name exp_result_output_dir
```

* or train a batch of experiments by modifying the argv list in worker.py, and execute
```
python worker.py

```

## Result Visualization
* Plot the experiment results in experiment result dir
```
python plot_result.py --csv_file result.csv --exp_dir models/exp_dir
```