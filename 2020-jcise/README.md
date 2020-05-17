This repository is provided for replication of results presented in:

A. Dourado and F. A. C. Viana, "[Physics-informed neural networks for missing physics estimation in cumulative damage models: a case study in corrosion fatigue](https://asmedigitalcollection.asme.org/computingengineering/article-abstract/doi/10.1115/1.4047173/1083614/Physics-informed-neural-networks-for-missing)," ASME Journal of Computing and Information Science in Engineering, Online first, 2020. (DOI: 10.1115/1.4047173).

This repository includes the following:
- **corrosion_fatigue_model.py**: contains function that creates the proposed recurrent neural network.
- **run00_MLP_m_training.py**: file that initializes the multilayer perceptron for corrosion-fatigue delta damage estimation.
- **run01_rnn_training.py**: trains the proposed recurrent neural network using the pre-trained multilayer perceptron
- **run02_rnn_prediction.py**: predicts the crack length in the wing panels of the entire fleet at the 7 year (or 20.000 flights) of operation. 

In order to run the codes, you will need to:
1. Install the PINN python package: https://github.com/PML-UCF/pinn
2. Download the datasets at: to be added.

## Disclosure:
To be added.

## Citing this repository

Please, cite this repository using: 

    @misc{2019_dourado_viana_python_corrosion_fatigue,
        author    = {A. Dourado and F. A. C. Viana},
        title     = {Python Scripts for Physics-Informed Neural Networks for Corrosion-Fatigue Prognosis},
        month     = Aug,
        year      = 2019,
        doi       = {10.5281/zenodo.3355729},
        version   = {0.0.1},
        publisher = {Zenodo},
        url       = {https://github.com/PML-UCF/pinn_corrosion_fatigue}
        }
The corresponding reference entry should look like:

A. Dourado and F. A. C. Viana, Python Scripts for Physics-Informed Neural Networks for Corrosion-Fatigue Prognosis, v0.0.1, Zenodo, https://github.com/PML-UCF/pinn_corrosion_fatigue, doi:10.5281/zenodo.3355729.
