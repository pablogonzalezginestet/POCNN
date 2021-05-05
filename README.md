# PO_CNN
 A Convolutional Neural Netwok approach for predicting cumulative incidence based on pseudo-observations

![](figure/2ndstage.png)

## Requirements
Install the following R packages: 

`utils.install_packages('prodlim')`

`utils.install_packages('eventglm')`

We use the functions `pseudo_coxph` from eventglm  and  `prodlim` and `jackknife` from prodlim.

## Simulations

The folder named Simulations contains the files to replicate the simulations.
The dataset CIFAR10 will be saved in the current directory under the folder name `./dataset_cifar`. You can speficy the directory passing the argument  `--dir`

To train and evaluate IPCW-PO-CNN single output :
```sh
python train_eval_pocnn.py --sample_size 1000 --nsim 50 --niter 25 --case 5 --po 'ipcwpo'
```

To train and evaluate IPCW-PO-CNN multi-output :
```sh
python train_eval_pocnn_multioutput.py --sample_size 1000 --nsim 50 --niter 25 --case 5 --po 'ipcwpo'
```

To train and evaluate Cox-PO-CNN :
```sh
python train_eval_coxcnn.py --sample_size 1000 --nsim 50 --niter 25 --case 5
```

Common **arguments:**

* `--lr`: learning rate (default = 0.0001)

* `--sample_size`: sample size (default=1000)

* `--nsim`: number of simulated data (default=100)

* `--case`: any case/scenario considered in the paper, there are six cases (default=1)

* '--data_dir': directory to save CIFAR10 dataset (default='./dataset_cifar')

Specific to PO and IPCW-PO:

* '--po': data generation using PO or IPCW-PO, it takes 'po' or 'ipcwpo' (default='po')

## Real data application

