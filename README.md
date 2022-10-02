We provide code to reproduce all the experiments presented in "Cross-Study Replicability in Cluster Analysis" (https://arxiv.org/pdf/2202.01910.pdf). 

# Structure

The repository is divided into 2 main folders:
* `synthetic_data/` which contains all the python scripts and data to replicate the analysis on synthetic data. 
* `real_data/` which contains all the python scripts and data to replicate the analysis on real cancer data. 

```
.
|____synthetic_data
| |____Appendix_Additional_Experiments.ipynb
| |____Appendix_Competing_Prediction_Accuracy.ipynb
| |____Appendix_Competing_Stability.ipynb
| |____Appendix_Competing_Tests.ipynb
| |____Plots.ipynb
| |____Fit.ipynb
| |____utils_synthetic.py
| |____Results
| |____Data
| | |____Gaussians
| | |____Shapes
| |____Plots
|____real_data
| |____Data_Processing.ipynb
| |____Fit.ipynb
| |____Plots.ipynb
| |____utils_real_data.py
| |____Data_Processing.R
| |____Data
| |____Results
```


# Data description

The synthetic data used can be found in `synthetic_data/Data` (`Gaussians` contains high dimensional Gaussian data, while `Shapes` contains the benchmark datasets from P. Fränti and S. Sieranoja, "K-means properties on six clustering benchmark datasets", Apllied Intelligence 2018).

The real data used can be found in `synthetic_data/r_data`, where the Mainz, Transbig and VDX cancer data can be found. 


# Fitting

In `Cancer/`, `gnomAD/` and `Synthetic/` you will find `Fit.ipynb`, an iPythonNotebook which contains all the code needed in order to fit the experiments and save the data necessary to then reproduce the plots. Notice: `Synthetic/Fit.ipynb` also contains code to produce figures for the syntetic data. The relevant functions called to fit the methods can be found in the `utils/` folder.

# Plotting

In `Cancer/` and `gnomAD/` you will find `Plots.ipynb`, an iPythonNotebook which contains all the code needed in order to produce the plots displayed in the paper.
* `Cancer/Plots.ipynb` reproduces in the main text (Figures 1 -- 5).
* `Synthetic/Fit.ipynb` reproduces in Appendices F, G (Figures 6 -- 20).
* `gnomAD/Plots.ipynb` reproduces in Appendix H (Figures 21 -- 38).
