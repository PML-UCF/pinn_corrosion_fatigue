# Corrosion-fatigue Crack Propagation with Physics-informed Neural Networks

This repository is provided for replication of results presented in the article:

A. Dourado and F.A.C. Viana, "Physics-Informed Neural Networks for Corrosion-Fatigue Prognosis", Proceedings of the Annual Conference of the PHM Society 2019

This repository includes the following sets of the code under PHM 2019:

corrosion_fatigue_model: contain the proposed RNN framework.

run00_MLP_logC_training: file that trains the MLP for the logC estimation.

run00_MLP_m_training: file that trains the MLP for Walker's model exponent (m) estimation.

run01_rnn_training: trains the proposed RNN using the pre-trained MLPs

run02_rnn_prediction: predicts the crack length in the wing panels of the entire fleet at the 7 year (or 20.000 flights) of operation. 


In order to run the codes, you will need to:

Install the PINN python package: https://github.com/PML-UCF/pinn

Download the datasets at: De Piemonte Dourado, Arinan; Viana, Felipe, 2019, "Replication Data for: PINN corrosion-fatigue_PHM 2019", https://doi.org/10.7910/DVN/PZ7DRJ, Harvard Dataverse, V1, and extract files to the directory where this repository is cloned.

Disclosures:

In the paper, we described a procedure for the MLPs training initialization revolving around random planes. For simplicity this procedure is not reproduced here, we rather opt to use an illustrative plane whose data values are expressed in the "MLP_training_data.csv" and "MLP_val_data.csv" files. 

The goal in this example is to illustrate the RNN presented in the referred citation (Dourado and Viana PHM 2019). We didn't pursue in this example optimizing the training parameters (MLP configureation, learning-rate, number of epochs) to accurately reproduce the results illustrated in the manuscript. But, we strongly encourage the user to experiment with the code in order to reproduce or even surpass the mentioned results.  
