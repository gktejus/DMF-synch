# Rotation Synchronization via Deep Matrix Factorization

This is the official Code for the paper[
Rotation Synchronization via Deep Matrix Factorization](https://github.com/gktejus/DMF-Synch) presented at ICRA 2023. 

## Setup

Our current implementation is based off [Implicit Regularization in Deep Matrix Factorization
](https://github.com/roosephu/deep_matrix_factorization). Please use python 3.7 to run this code.

The basic libraries are listed in `requirements.txt`. You can install them by running the following command.

```bash
pip install -r requirements.txt
```

Another way to do this is by using the setup bash script. This will install the required libraries and download and arrange the datasets. This can be done by running the following command.

```bash
bash setup.sh
```     

## Datasets and Configs ( 1DSfM / Synthetic )

### 1DSfM 
For our experiments, we use the 1DSfM dataset and synthetic datasets. The 1DSfM dataset can be downloaded from [here](https://drive.google.com/file/d/1RTiEFxRK4ub4D-VlUkS9jX_N5AGT6VJF/view?usp=share_link).

The general structure of the datasets and configs for a particular dataset is as follows.

```
.
└── DMF-Synch
    ├── datasets
    │   ├── data-real
    │   │   ├── [dataset]_gt.pt ------> Ground truth relative rotations (noisy)
    │   │   ├── [dataset]_obs.pt -----> Observed indices
    │   │   └── [dataset]_unobs.pt ---> Unobserved indices
    │   └── data-real-mat
    │       └── [dataset].mat --------> Ground truth absolute rotations
    └── configs
        └── data-real
            └── [dataset].toml --------> Config file
```


The easiest way to setup the 1DSfM dataset is to run the bash script `setup.sh`. Please note that this will download and arrange all the 15 datasets contained in the 1DSfM dataset. The configs for each dataset in 1DSfM are already present in the configs folder. 

### Synthetic Datasets
To generate the synthetic datasets, you'll first have to generate the ground truths and the mask using the MATLAB script `synth_real.m` located in the `MATLAB-scripts` folder. 

This script has several paramerters that can be changed. The parameters are as follows:

- `ncams_c` : Number of cameras
- `prob_out` : Fraction of outliers
- `sigma_a` : Noise in degrees
- `fraction_missing` : Fraction of missing data


After setting the parameters and running the MATLAB script, to generate the final datasets and configs, please run the following command.

```python
python gen_synth.py 
```




## Experiments

If you just want to run one experiment, use the following command as an example. 

```bash
python3 main.py --print_config --log_dir ./logs \
    --config configs/mat-real/Alamo.toml, \
    --config configs/opt/SGD.toml, \
    --set depth 2, \
    --set lr 0.001, \
    --set initialization gaussian, \
    --set init_scale 1.e-3

```



To run multiple experiments with multiple hyperparameters sequentially, you can use `./scripts/run.rb` (please make sure Ruby is installed and `gem install colorize --user`). The code will log into `~/logs` by default. 



```bash
./scripts/run.rb --n_jobs 1 --name mat-cmpl --template 'python3 main.py \
    --print_config --log_dir ./logs/ --config configs/data-real/CONFIG, \
    --config configs/opt/SGD.toml, --set depth DEPTH --set lr LR \
    --set init_scale SCALE  --set n_iters 100000 --set initialization INIT'\
    --replace LR=0.3 \
    --replace DEPTH=2,3 \
    --replace SCALE=1.e-3 \
    --replace CONFIG=Alamo.toml, Ellis_Island.toml\ 
    --replace INIT=gaussian
```

