# DP_lazy_PI

## Description

This repository contains the code for the paper "Fast, Distribution-free Predictive Inference for Neural Networks with Coverage Guarantees". The paper introduces a novel, computationally-efficient algorithm for predictive inference (PI) that requires no distributional assumptions on the data and can be computed faster than existing bootstrap-type methods for neural networks.

## Installation

```shell
pip3 install -r requirements.txt
python3 main.py --help
```

## Usage

```plaintext
usage: main.py [-h] [--data_name DATA_NAME] [--epoch EPOCH] [--tot_trial TOT_TRIAL] [--alpha ALPHA] [--train_size TRAIN_SIZE]
               [--hidden_width HIDDEN_WIDTH] [--lam LAM] [--iseed ISEED] [--psim PSIM]

options:
  -h, --help            show this help message and exit
  --data_name DATA_NAME, -dn DATA_NAME
  --epoch EPOCH, -e EPOCH
                        number of epochs when training the neural networks
  --tot_trial TOT_TRIAL, -t TOT_TRIAL
                        total number of trials
  --alpha ALPHA, -a ALPHA
                        level of significance
  --train_size TRAIN_SIZE, -n TRAIN_SIZE
                        training size
  --hidden_width HIDDEN_WIDTH, -w HIDDEN_WIDTH
  --lam LAM, -l LAM     ridge penalty parameter 
  --iseed ISEED, -s ISEED
                        data generating random seed
  --psim PSIM, -p PSIM  simulation p
                        dimension of data features in the simulation
```

## Example
```
python3 main.py -e 20 -n 100 -p 16 -dn sim
```
or
```
module load python/python3.9.9 #if python3 is not available
python main.py -e 20 -n 100 -p 16 -dn sim
```

## License

The code is released under the MIT License.
