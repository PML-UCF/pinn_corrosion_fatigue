This repository is provided for replication of results presented in:

A. Dourado and F. A. C. Viana, "[Physics-informed neural networks for missing physics estimation in cumulative damage models: a case study in corrosion fatigue](https://asmedigitalcollection.asme.org/computingengineering/article-abstract/doi/10.1115/1.4047173/1083614/Physics-informed-neural-networks-for-missing)," ASME Journal of Computing and Information Science in Engineering, Online first, 2020. (DOI: 10.1115/1.4047173).

This repository includes the following:
- **bias_model.py**: contains function that generates the proposed hybrid recurrent neural network.
- **run00_MLP.py**: file that initializes multiple architectures for the multilayer perceptron considering several auxiliary planes. 
- **run01_rnn_training.py**: trains the proposed hybrid recurrent neural network considering the multiple pre-trained multilayer perceptrons.
- **run02_rnn_x_validation.py**: file that implements a cross-validation procedure to help selecting the most suitable combination of multilayer perceptron acrchitecture and auxiliary plane for damage prediction. 
- **run03_rnn_diagnosis.py**: predicts the crack length in the entire fleet at the 5th year (or 15,000 flights) of operation. 
- **run04_rnn_prognosis.py**: damage forecast of the entire fleet at the 6th year (or 18,000 flights) of operation. 

In order to run the codes, you will need to:
1. Install the PINN python package: https://github.com/PML-UCF/pinn
2. Download the datasets at: Dourado, Arinan; Viana, Felipe, 2020, "Replication Data for: ASME_JCISE_2020", https://doi.org/10.7910/DVN/OZJEZE, Harvard Dataverse, V1

## Disclosure:
The codes provided for training here ("run00_MLP", "run01_rnn_training", and "run02_rnn_x_validation") will implement all 140 possible combinations of multilayer perceptron and auxiliary plane described in the paper. These procedures are computationally intense and very time consuming.
Additionally, it is worth mentioning that the training procedures implemented in the above codes involve some random steps and results illustrated here may not perfectly reflect the results presented in the paper.
For simplicity, only the best configuration regarding multilayer perceptron architecture and auxiliary plane discussed in the paper were implemented in the damage forecast codes ("run03_diagnosis", and "run04_prognosis").
In these examples, we did not pursue optimizing any of the training parameters (multilayer perceptron configuration, learning-rate, number of epochs) to accurately reproduce the results illustrated in the manuscript. 
Nevertheless, we strongly encourage the user to experiment with the code in order to reproduce or even surpass the mentioned results.

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
