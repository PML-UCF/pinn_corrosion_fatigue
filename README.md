# Corrosion-fatigue Crack Propagation with Physics-informed Neural Networks

This repository is provided for replication of results presented in the article:

A. Dourado and F.A.C. Viana, "Physics-Informed Neural Networks for Corrosion-Fatigue Prognosis", Proceedings of the Annual Conference of the PHM Society 2019

This repository includes the following sets of the code under PHM 2019:

run00_MLP_data_generation generates a plane approximation for MLPs training.

run00_MLP_logC_training file that trains the MLP for the logC estimation.

run00_MLP_m_training file that trains the MLP for Walker's model exponent (m) estimation.

run01_rnn_training trains the proposed RNN using the pre-trained MLPs

run02_rnn_prediction predicts the crack length in the wing panels of the entire fleet at the 7 year (or 20.000 flights) of operation. 


In order to run the codes, you will need to:

Install the PINN python package: https://github.com/PML-UCF/pinn

Download the datasets at: De Piemonte Dourado, Arinan; Viana, Felipe, 2019, "Replication Data for: PINN corrosion-fatigue_PHM 2019", https://doi.org/10.7910/DVN/PZ7DRJ, Harvard Dataverse, V1, and extract files to the directory where this repository is cloned.
