We provide code for clustering replicability analyses. The code provided here allows to replicate all the experiments in https://arxiv.org/pdf/2202.01910.pdf

```
.
|____synthetic_data
| |____Appendix_Competing_Prediction_Accuracy.ipynb
| |____final_synthetic_experiments.ipynb
| |____utils_synthetic.py
| |____Appendix_Competing_Stability.ipynb
| |______pycache__
| | |____utils_synthetic.cpython-38.pyc
| |____Results
| | |____test_set_3.npy
| | |____.DS_Store
| | |____score_ami_3.npy
| | |____score_ari_3.npy
| | |____train_set_3.npy
| | |____special_points_3.npy
| | |____exercise_1.npy
| | |____exercise_2.npy
| |____final_appendix.ipynb
| |____final_main_paper_plots.ipynb
| | |____final_main_paper_plots-checkpoint.ipynb
| | |____Appendix_Competing_Tests-checkpoint.ipynb
| | |____final_synthetic_experiments-checkpoint.ipynb
| | |____final_appendix-checkpoint.ipynb
| |____Additional_Experiments
| | |____calibration_ami_boot.npy
| | |____jaccard_results.npy
| | |____prediction_test.npy
| | |____no_clusters_ari.npy
| | |____dependent_ami.npy
| | |____.DS_Store
| | |____calibration_ami_MC_2.npy
| | |____mini_test_scores.npy
| | |____calibration_ari_MC_2.npy
| | |____calibration_ari_boot_2.npy
| | |____dependent_ari.npy
| | |____calibration_ami_boot_2.npy
| | |____mini_test_scores_2.npy
| | |____no_clusters_ami.npy
| | |____calibration_train.npy
| | |____mini_test_scores_distance.npy
| | |____no_clustering_scaling_d.pdf
| | |____calibration_ari_boot.npy
| | |____no_clusters_ami_scaling_d.npy
| | |____calibration_test.npy
| | |____d_r_scores_results.npy
| | |____calibration_test_2.npy
| | |____calibration_ami_MC.npy
| | |____smolkin_results.npy
| | |____no_clusters_ari_scaling_d.npy
| | |____calibration_train_2.npy
| | |____calibration_ari_MC.npy
| |____Appendix_Competing_Tests.ipynb
| |____Data
| | |____Gaussians
| | | |____dim512.txt
| | | |____dim128.txt
| | | |____dim064.txt
| | | |____dim1024.txt
| | | |____dim256.txt
| | | |____dim032.txt
| | |____Shapes
| | | |____d31.txt
| | | |____r15.txt
| | | |____pathbased.txt
| | | |____compound.txt
| | | |____spiral.txt
| | | |____jain.txt
| | | |____flame.txt
| | | |____aggregation.txt
| |____Plots
| | |____Test_k2.pdf
| | |____stability_r_aggregation.pdf
| | |____stability_d_r15.pdf
| | |____Test_k1.pdf
| | |____ex_2_032.pdf
| | |____no_clustering.pdf
| | |____.DS_Store
| | |____stability_d_compound.pdf
| | |____calibration_2.pdf
| | |____ex_2_1024.pdf
| | |____stability_r_r15.pdf
| | |____ex_2_256.pdf
| | |____calibration_quantiles.pdf
| | |____no_clustering_scaling_d.pdf
| | |____exercise_1.pdf
| | |____calibration_ds.pdf
| | |____ind_cl.pdf
| | |____ex_1.pdf
| | |____calibration_quantiles_2.pdf
| | |____ex_2_064.pdf
| | |____stability_d_aggregation.pdf
| | |____ex_2_128.pdf
| | |____ex_2_512.pdf
| | |____main_synth_3_b.pdf
| | |____Test_distance.pdf
| | |____n_dependence_ds.pdf
| | |____n_dependence.pdf
| | |____stability_r_compound.pdf
| | |____calibration.pdf
| | |____calibration_ds_2.pdf
|____real_data
| |____data_processing.ipynb
| |____r_data
| | |____MAINZ_3_SV.csv
| | |____transbig_subtypes.csv
| | |____transbig_dict.npy
| | |____vdx_dict.npy
| | |____vdx_subtypes.csv
| | |____tSNE_embedding_transbig.npy
| | |____VDX_3_SV.csv
| | |____TRANS_3_SV.csv
| | |____mainz_subtypes.csv
| | |____mainz_dict.npy
| |______pycache__
| | |____utils_real_data.cpython-38.pyc
| |____real_data_plots.ipynb
| |____results
| | |____score_ari_local.npy
| | |____score_ami_local.npy
| | |____scores_labels.npy
| |____utils_real_data.py
| |____.ipynb_checkpoints
| | |____checks-checkpoint.ipynb
| | |____real_data_plots-checkpoint.ipynb
| | |____data_processing-checkpoint.ipynb
| | |____Untitled-checkpoint.ipynb
| | |____real_data_experiments-checkpoint.ipynb
| |____real_data_experiments.ipynb
| |____data_processing.R
```