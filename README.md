# PO_CNN
 A CNN approach for predicting cumulative incidence based on pseudo-observations

## Simulations

The folder named Simulations contains the files to replicate the simulations.
The dataset CIFAR10 will be saved in the current directory under the folder name `./dataset_cifar`. You can speficy the directory passing the argument  `--dir`
For example:
```sh
python train_evaluate.py --sample_size 1000 --nsim 50 --niter 25 --case 5 --po 'ipcwpo'
```
