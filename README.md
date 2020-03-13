@Author Marek Szakacs, University of Innsbruck, 2020
This code is part of a bachelor thesis: Neural Turing Machines for modelling attention.

NTM implementation was taken from https://github.com/wchen342/NeuralTuringMachine

To execute a single training run: 
python run_tasks_single_parameters.py --<argument> <value> --<argument> <value> ... --<argument> <value>
Example: python run_tasks_single_parameters.py --num_epochs 10 --dataset_dir dataset/1

To execute multiple training runs on a single dataset with different hyperparameters using HParams run:
python run_tasks_single_parameters.py --<argument> <value> --<argument> <value> ... --<argument> <value>
Example: python run_tasks_single_parameters.py --num_epochs 10 --dataset_dir dataset/1

To view TensorBoard logs execute following command:
tensorboard --logdir <dir_name>
Example: tensorboard --logdir logs

