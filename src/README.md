# For MS-CMRSeg data set

## Train
    run train_mscmrseg.py. to start training 
    Example: python train_mscmrseg.py -aug2 -bs 16 -ns 2000 -e 400 -d1 -d2 -d4 -data_dir ../../input_aug/

* Please specify the data directory to the data set and specify the discrimiantors by adding arguments 'd1', 'd2' or 'd4'.

## Evaluate
* run evaluate_mscmrseg.py. to start evaluation.
* Please change the directory to the weights before evaluation.

## Data set
* to generate point cloud ground truth, please follow the function npy2point() in ./utils/npy2point.py.

# For MM-WHS data set

## Train

    run train_mmwhs.py to start training
    Example: python train_mmwhs.py -bs 16 -ns 1000 -e 400 -data_dir ../../input/ -d1 -d2 -d4 -offdecay -lr_fix 0.0002 -lr 0.0002 -d1lr 1e-04 -d2lr 5e-05 -d4lr 1e-04 -dr 1 -ft -extd4
* After training, the model with the highest Dice Similarity Coefficient will be evaluated automatically.
* To reproduce the result reported in the paper, please follow the following learning configuration:


| D1            | D2            | D3    | Learning rate | &#955   |
|:-------------:|:-------------:|:-----:|:-------------:|:-------------:|
|:heavy_check_mark:| right-aligned | $1600 |
| col 2 is          | centered      |   $12 |
| zebra stripes     | are neat      |    $1 |


## Evaluate
* run evaluate_mmwhs.py to start evaluation.
* Please change the directory to the weights before evaluation.

## Data set
* To convert tfrecords files to numpy array, please use ./utils/tf_to_numpy.py.
* Please change the directory of the data before running. 
