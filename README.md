We provide code to reproduce all the experiments presented in "Cross-Study Replicability in Cluster Analysis" (Statist. Sci. 38(2): 303-316 (May 2023). DOI: 10.1214/22-STS871)[https://projecteuclid.org/journals/statistical-science/advance-publication/Cross-Study-Replicability-in-Cluster-Analysis/10.1214/22-STS871.short]. See also https://arxiv.org/pdf/2202.01910.pdf). 

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

In both `synthetic_data/` and `real_data/` the iPython notebook `Fit.ipynb` contains all the code needed in order to fit the experiments and save the data necessary to then reproduce the plots in the main text.

# Plotting

In both `synthetic_data/` and `real_data/` the iPython notebook `Plots.ipynb` contains all the code needed in order to reproduce the plots in the main text. The intermediary files accessed are stored in the `Results/` folder.

# Appendix

To replicate the experiments in the Appendix, run the `synthetic_data/Appendix_*` iPython notebooks. These notebooks contain both code to run (i) generate synthetic data, (ii) run the analyses and (iii) reproduce plots.
