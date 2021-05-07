# PO_CNN
## A Convolutional Neural Netwok approach for predicting cumulative incidence based on pseudo-observations

![](figure/2ndstage.png)

## Requirements
Install the following R packages: 

`utils.install_packages('prodlim')`

`utils.install_packages('eventglm')`

We use the functions `pseudo_coxph` from eventglm  and  `prodlim` and `jackknife` from prodlim.

## Simulations

The folder named Simulations contains the files to replicate the simulations.
The dataset CIFAR10 will be saved in the current directory under the folder name `./dataset_cifar`. You can speficy the directory passing the argument  `--dir`

Common **arguments:**

* `--lr`: learning rate (default = 0.0001)

* `--sample_size`: sample size (default=1000)

* `--nsim`: number of simulated data (default=100)

* `--case`: any case/scenario considered in the paper, there are six cases (default=1)

* `--data_dir`: directory to save CIFAR10 dataset (default='./dataset_cifar')

Specific to PO and IPCW-PO:

* `--po`: data generation using PO or IPCW-PO, it takes 'po' or 'ipcwpo' (default='po')

For example:

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


## Real data application
The folder named real_data_application contains the files to run PO-CNN, IPCW-PO-CNN and Cox-CNN.

The file TCGA_clinical.csv contains the clinical data (structured data). The column `Test` indicates the patients that we consider in the training set and test set.

The file clinical_data.py contains different functions to format the data corresponding to the model:
- `get_clinical_data_po` pre-processes the structured data to be used in PO-CNN
- `get_clinical_data_ipcwpo` pre-processes the structured data to be used in IPCW-PO-CNN
- `get_clinical_data_cox` pre-processes  the structured data to be used in Cox-CNN


### Images 
The images can be downloaded from [GDC data transfer API](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool).

#### Preprocessing
Each WSI was preprocessed before inclusion into this study. As a first step, tissue masks were generated. To this end, WSIs were downsampled by a factor of 32 and converted to the HSV color space. Tissue masks were generated by applying a pixel-wise logical “and” operation between a mask that was generated by applying the Otsu threshold (ref1) to the saturation channel and by applying a cutoff of 0.75 to the hue channel. We subsequently performed morphological opening and closing to remove salt-and-pepper noise from the binary masks. WSIs were then tiled with 50% overlap at 20X magnification into image patches spanning 598x598 pixels, where 598 pixels correspond to 271µm of tissue section, while discarding tiles with less than 50% tissue content. Tiles were then color-normalized using the method described by Macenko et al. (ref2). We subsequently applied a cancer detection CNN to identify cancer regions and excluded all tiles that were predicted to belong to a benign tissue region. Furthermore, out-of-focus tiles with a low variance were excluded, which was computed by filtering each tile with a Laplacian. 

### Training, Evaluating and Test

To train, validate and test PO-CNN single-output (default arguments) :
```sh
python train_eval_test_pocnn.py 
```
To train, validate and test IPCW-PO-CNN single-output (default arguments) :
```sh
python train_eval_test_pocnn.py --po 'ipcwpo' 
```
To train, validate and test PO-CNN multi-output (default arguments) :
```sh
python train_eval_test_pocnn.py  --implementation 'multi_output'
```

**arguments:**

* `--max_num_epochs`:  number of epochs (default=30)

* `'--num_samples'`: number of times to sample from the hyperparameter space  (default=3)

* `---gpus_per_trial`: gpu resources to be used per trial (default=1)

* `--cpus_per_trial`: gpu resources to be used per trial (default=4)

* `--po_cnn`: 'po' or 'ipcwpo' (default='po')

* `--implementation`: to run PO-CNN using the approach single output ('single_output') or multi output ('multi_output') (default= 'single_output')

* `--data_dir_train`: directory where the train images are stored (default = './data/Macenko_new_normed_TCGA_size_1196_stride_598_resize_1' )

* `--data_dir_test`: directory where the test images are stored (default = './data/test_set/Macenko_new_normed_TCGA_size_1196_stride_598_resize_1')


