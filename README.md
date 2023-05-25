# Phos_drp_example
Drug response prediction (DRP) using phosphoproteomics data

## Quickstart 
See notebooks/1.0-explained-phos-DRP for a walkthrough of drug response prediction using phosphoproteomics data. This notebook applies the code within the src folder.

The src folder contains modules for data loading, preprocessing model creation model training and model evaluation. 


## Problem Formulation 

The goal of DRP is to predict how effective different drugs are for different cancer types. 
Here we predict the I50 values, the concentration of a drug needed to inhibit the activity of a cell lie by 50%, as a measure of efficacy. 
We feed phosphoproteomics profiles of cell lines and simple column representaions of drugs though a neural network to do this. 

Consider the traning set $T = \{ \boldsymbol{x_{c,i}}, \boldsymbol{x_{d,i}}, y_i\} $ where 
$\boldsymbol{x_{c,i}}$, $\boldsymbol{x_{d,i}}$  are representation of the $i^{th}$ cell line and drug respectively and
 $y_i$ is the IC50 value associated with the $i^{th}$ cell line drug pair.

 Thus, we want to find a model, $M$, that takes $\boldsymbol{x_{c,i}}$ and $\boldsymbol{x_{d,i}}$ as inputs and predicts for the corresponding IC50 value $\hat{y_i}$ such that $M(\boldsymbol{x_{c,i}}, \boldsymbol{x_{d,i}})=\hat{y_i}$.

We test the model on what is known as cell blind testing. This means that none of the cell lines in the testing set are in the training set. Cell blind testing simulates how a model would perform in a stratified medicine context. Where previous patents' responses to a set of drugs would be available for training and the goal would be to predict the response of a new patent to these drugs. 