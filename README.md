### Neural Turing Machines for modelling attention.

NTM implementation was taken from: https://github.com/wchen342/NeuralTuringMachine\
Dataset was generated using Fleuret framework: https://fleuret.org/git-tgz/svrt\
This repository contains only images of problem 1. For generation of other problems, use the framework mentioned above.

To execute a single training run:
```console
python run_tasks_single_parameters.py --<argument> <value> --<argument> <value> ... --<argument> <value>
Example: python run_tasks_single_parameters.py --num_epochs 10 --dataset_dir dataset/1
```
To execute multiple training runs on a single dataset with different hyperparameters using HParams run:
```console
python run_tasks_single_parameters.py --<argument> <value> --<argument> <value> ... --<argument> <value>
Example: python run_tasks_single_parameters.py --num_epochs 10 --dataset_dir dataset/1
```
To view TensorBoard logs execute following command:
```console
tensorboard --logdir <dir_name>
Example: tensorboard --logdir logs
```
This code is a part of bachelor thesis: Neural Turing Machines for modelling attention.\
University of Innsbruck, 2020.
