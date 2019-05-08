# Deep Sky

Deep Sky is a package for performing deep learning tasks on meteorological problems. The 
package contains programs for training and evaluating models that predict the probability
of hail. There are also programs for generating random fields using generative adversarial 
networks.

The paper describing the code in this library is
* Gagne, D. J., S. E. Haupt, D. W. Nychka, and G. Thompson, 2019: Interpretable Deep Learning for Spatial Analysis of Severe Hailstorms. Monthly Weather Review.

## Installation

The following libraries are required for installation and can be installed using 
either conda or pip.
* numpy
* scipy
* matplotlib
* xarray
* netcdf4
* tensorflow
* keras
* scikit-learn
* pandas
* numba
* pyyaml

Deepsky can be downloaded from Github at https://github.com/djgagne/deepsky. Once
downloaded, the software can be installed with the following command:
```
$ pip install .
```

## Usage

All models can be trained and evaluated using the scripts in the scripts directory.

## Data

The data for the project can be found on the NCAR GLADE system at `/glade/work/dgagne/interpretable_deep_learning_data/`. There are 2 directories of files.

1. `ncar_ens_storm_patches`: This directory contains netCDF files from the NCAR Convection Allowing ensemble. Each file contains multi-variable 32x32 grid cell patches
surrounding each storm in the model. The files are formatted `ncar_ens_storm_patches_YYYYMMDDHH_mem_NN.nc` where the first set of numbers is the model run initial date, and the 
second number is the ensemble member.

2. `spatial_storm_results_20171220`: This directory contains verification statistics in csv files, trained keras neural network model files in HDF5 (h5) format, and trained scikit-learn models
in pickle (pkl) file format. Each machine learning and deep learning model was cross-validated for hyperparameter tuning and trained over 30 separate iterations. The model number is the 3-digit numbrer at the end of the filename. Each iteration uses different run dates for the training and testing sets.

The csv files in this directory as categorized as follows:
* `model_param_combos.csv`: the hyperparameter settings for that model that are tested.
* `model_best_params.csv`: the best hyperparameters from each trial.
* `model_param_scores_sample_000.csv`: The verification scores for each hyperparameter setting.
* `model_sample_scores.csv`: The verification scores on the testing set for each of the 30 iterations.
* `var_importance_model_score_000.csv`: The permutation variable importance scores for a particular trial.
## 

## Analysis
The analysis for the paper can be found in Jupyter notebooks in the notebooks directory. The most relevant notebooks are
1. `ThompsonHail.ipynb`: Used to generate Figure 1
2. `radar_hail_cnn.ipynb`: Used to generate Figure 2
3. `spatial_hail_feature_importance.ipynb`: Plotting feature importances for each model. 
4. `SpatialHailInterp.ipynb`: Verification and plotting of ideal hailstorms from Conv. Net.
5. `Logistic_PCA_Interp.ipynb`: Plot of ideal hailstorms from Logistic Mean and Logistic PCA models.
6. `Hail_Conv_Net_Filter_Activations.ipynb`: Used to generate Figures 11-13.

