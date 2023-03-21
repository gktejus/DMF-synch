# Implicit Regularization in Deep Matrix Factorization

Code for [
Implicit Regularization in Deep Matrix Factorization](https://arxiv.org/abs/1905.13655). 

## Installation

Please ues Python 3.7 for running this code. 

```bash
pip install -r requirements.txt
```

## Dataset Generation

Here is the example for generating the inputs for matrix completion with n = 100, rank = 5 and 2k samples. 

```bash
mkdir -p datasets/mat-cmpl
python gen_gt.py --config configs/mat-cmpl/gen_gt,
python gen_obs.py --config configs/mat-cmpl/gen_obs, --set n_train_samples 2000
```

## Experiments

If you just want to run one experiment, use the following command as an example. 

```bash
python main.py --print_config --log_dir /tmp/exp1 \
    --config configs/mat-cmpl/run, \
    --config configs/mat-cmpl/2000, \
    --config configs/opt/grouprmsprop, \
    --set depth 2 
```

For nuclear norm minimization: 

```bash
python main.py --print_config --log_dir /tmp/exp2 \
    --config configs/mat-cmpl/run, \
    --config configs/mat-cmpl/2000, \
    --config configs/opt/cvx,
```

For dynamics of gradient descent (Figure 3):

```bash
python main.py --log_dir /tmp --print_config \
    --config configs/ml-100k, \
    --config configs/opt/SGD, \
    --config configs/dynamics, \
    --set depth 2
```
python main.py --print_config --log_dir "./test" --config configs/opt/grouprmsprop, --config configs/


The results will be saved at `/tmp/ID`, where `ID` is a different number for each run and startsfrom 0.  

To run multiple experiments sequentially, you can use `./scripts/run.rb` (please make sure Ruby is installed and `gem install colorize --user`). The code will log into `~/logs` by default. 

```bash
./scripts/run.rb --n_jobs 3 --name mat-cmpl \
    --template 'python main.py --print_config --log_dir LOGDIR --config configs/mat-cmpl/run, --config configs/mat-cmpl/SAMPLES, --config configs/opt/grouprmsprop, --set depth DEPTH --set lr LR --set init_scale SCALE' \
    --replace LR=0.001,0.0003 \
    --replace DEPTH=2,3,4 \
    --replace SCALE=1.e-3,1.e-4,1.e-5,1.e-6 \
    --replace SAMPLES=2000,5000
```

For multiple experiments on nuclear norm minimization: 

```bash
./scripts/run.rb --n_jobs 1 --name mat-cmpl-cvx \
    --template 'python main.py --print_config --log_dir LOGDIR --config configs/mat-cmpl/run, --config configs/mat-cmpl/SAMPLES, --config configs/opt/cvx,' \
    --replace SAMPLES=2000,5000
```
# Plotting

We use the Jupyter notebook `plot.ipynb` to generate our figures. 

Please modify 4-th cell to load all results. The directories are the corresponding `--log_dir` option, e.g., `/tmp/exp1` in the first example. 
 

./scripts/run.rb --n_jobs 1 --name mat-cmpl \
    --template 'python main.py --print_config --log_dir ./logs/ --config configs/mat-cmpl/CONFIG, --config configs/opt/SGD, --set depth DEPTH --set lr LR --set init_scale SCALE --set seed 42  --set n_iters 100000 --set initialization INIT --set project_name init_var' \
    --replace LR=0.3 \
    --replace DEPTH=5 \
    --replace SCALE=1.e-3 \
    --replace CONFIG=Ellis_Island,Alamo,Gendarmenmarkt,Madrid_Metropolis,Montreal_Notre_Dame,Notre_Dame,NYC_Library,Piazza_del_Popolo,Piccadilly,Roman_Forum,Tower_of_London,Trafalgar,Union_Square,Yorkminster\
    --replace INIT=identity,orthogonal,uniform


python main.py --print_config --log_dir ./test/ --config configs/opt/grouprmsprop, --config configs/mat-cmpl/NYC_Library,  --set depth 2 --set init_scale 1.e-3 --set lr 0.0003  --set seed 42

./scripts/run.rb --n_jobs 1 --name mat-cmpl \
    --template 'python3 main.py --print_config --log_dir ./logs/ --config configs/mat-synth/CONFIG, --config configs/opt/SGD, --set depth DEPTH --set lr LR --set init_scale SCALE --set seed 42  --set n_iters 100000 --set initialization INIT --set project_name synthetic_experiments --set is_synthetic True' \
    --replace LR=0.3 \
    --replace DEPTH=2,3,4,5 \
    --replace SCALE=1.e-3 \
    --replace CONFIG=test_CAM_100_Out_0.4_miss_0.4,test_CAM_200_Out_0.4_miss_0.6,test_CAM_400_Out_0.4_miss_0.8,test_CAM_100_Out_0.4_miss_0.5,test_CAM_200_Out_0.4_miss_0.7,test_CAM_400_Out_0.4_miss_0.9,test_CAM_100_Out_0.4_miss_0.6,test_CAM_200_Out_0.4_miss_0.8,test_CAM_600_Out_0.4_miss_0.4,test_CAM_100_Out_0.4_miss_0.7,test_CAM_200_Out_0.4_miss_0.9,test_CAM_600_Out_0.4_miss_0.5,test_CAM_100_Out_0.4_miss_0.8,test_CAM_400_Out_0.4_miss_0.4,test_CAM_600_Out_0.4_miss_0.6,test_CAM_100_Out_0.4_miss_0.9,test_CAM_400_Out_0.4_miss_0.5,test_CAM_600_Out_0.4_miss_0.7,test_CAM_200_Out_0.4_miss_0.4,test_CAM_400_Out_0.4_miss_0.6,test_CAM_600_Out_0.4_miss_0.8,test_CAM_200_Out_0.4_miss_0.5,test_CAM_400_Out_0.4_miss_0.7,test_CAM_600_Out_0.4_miss_0.9 \ 
    --replace INIT=gaussian


./scripts/run.rb --n_jobs 1 --name mat-cmpl \
    --template 'python3 main.py --print_config --log_dir ./logs/ --set add_reg REG --config configs/mat-test/CONFIG.toml --config configs/opt/SGD.toml --set depth 5 --set lr 0.3 --set init_scale 1e-3 --set seed 42  --set n_iters 100000 --set initialization gaussian --set project_name reg_search_rest --set is_synthetic False' \
    --replace CONFIG=Alamo,Madrid_Metropolis,Notre_Dame,Roman_Forum,Union_Square,Ellis_Island,Montreal_Notre_Dame,Piazza_del_Popolo,Tower_of_London,Vienna_Cathedral,Gendarmenmarkt,Piccadilly,Trafalgar,Yorkminster\
    --replace REG=0.1,0.3,0.7,1e-3,3e-3,7e-3,1e-2,3e-2,7e-2,1e-4,3e-4,7e-4
