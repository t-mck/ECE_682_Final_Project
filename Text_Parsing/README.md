# README

#### Start with the dnn_main.py file. This file will look the most like a standard analysis script; meaning that it:
1. Loads old data, that has predictive variables and a response variable
2. Builds a model with that old data
3. Loads new data, that only has predictive variables
4. Generates predictions for that new data using the model
5. Explores those predictions using standard approaches like summary statistics, histograms, etc.

#### Directory Explanation

1. data_code
   -  Contains code that is specific to loading, cleaning, and generating data used to build a model

2. model 
   -  Contains code that is specific to building neural network models, and exploring the training 
progress of those models

3. predictions (currently empty)
   -  Contains code that is specific to exploring the predictions of a model after they have been 
generated. This is just provided as an example, feel free to use your own functions, or functions from another package.
