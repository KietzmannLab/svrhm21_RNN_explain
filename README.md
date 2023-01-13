Code and data accompanying our SVRHM'21 paper, "Category-orthogonal object features guide information processing in recurrent neural networks trained for object categorization" - https://arxiv.org/abs/2111.07898

Requires tensorflow 1.13, python 3.7, scikit-learn, and pytorch 1.6.0 to be installed.

Python scripts included:
1. RNN_gen.py can be used to train the RNNs.
2. RNN_analyse_reprs_recurrence.py can be used to train linear classifiers for auxiliary variables and category decoding, on the layer activations and recurrent flows
3. RNN_perturb.py can be used to perform the perturbation analyses.

Two pretrained RNNs can be downloaded from https://osf.io/pf4u5/ to be assessed using the jupyter notebook 'SVRHM21_results.ipynb'. Alternatively, you can train your own RNN and assess it.
