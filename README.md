# Artifacts for paper: *Differential Privacy of Quantum and Quantum-Inspired Classical Recommendation Algorithms*

We mainly use Python 3.10.9 with `numpy` and `scipy` to run experiments and `matplotlib` to draw the results. The experiments are run on a PC with Intel Core i7-8750H CPU and 16GB memory and the code should be compatible with any recent versions.

The MovieLens datasets can be downloaded from [https://files.grouplens.org/datasets/movielens/](https://files.grouplens.org/datasets/movielens/), and the Netflix dataset can be downloaded from Kaggle. Our code is available via the anonymous link [https://anonymous.4open.science/r/paper-DP-for-qRS_artifacts-82B0](https://anonymous.4open.science/r/paper-DP-for-qRS_artifacts-82B0) (this site).

The components of the anonymous repository are as follows:

+ `README.md`: This readme file.
+ `incoherence_demo.ipynb`: Jupyter notebook that checks the incoherence of MovieLens datasets and plots Fig. 2 and Fig. 5.
+ `exp_evaluation.ipynb`: Jupyter notebook that conducts the experiments and evaluation in Sec. 7.1 and computes Table 1.
+ `privacy_matched_noise_level_comparison.ipynb`: Jupyter notebook that plots Fig. 4.
+ `k_dependence.ipynb`: Jupyter notebook that conducts the experiments in Appendix E and plots Fig. 6.
+ `data_loader.py`: Data loader that loads certain MovieLens dataset into matrix form. The datasets need to be in the same folder as the loader script.