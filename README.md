# Project
State shortly what you did during each week. Just a table with the main results is enough. Remind to upload a brief presentation (pptx?) at virtual campus. Do not modify the previous weeks code. If you want to reuse it, just copy it on the corespondig week folder.

## Task1 work
The purpose of this week's task is to perform a study to tune a series of parameters that compose a BoVW algorithm which is used to correctly classify some given images into each one of the 8 different classes that form the given dataset. After the study has been performed, this combination of parameters has shown the best results:

| Parameter           | Best performance | Addition                  | 
| :---                |    :----:        |     :----:                  |
| Descriptor          | Dense SIFT       | -                         |
| Scale               | 1                | -                         |
| Spatial Pyramid     | True             | 3 levels                  |
| Normalization       | True             | Scaler                    |
| Codebook            | 160              | -                         |
| Classifier          | SVM              | Histogram intersection    |
| Dimensionality      | None              | -                         | 


The test split has been evaluated with this configuration, yielding the following results:

| Split      | Accuracy    | Precision   | Recall      | F-Score     |
| :---       |    :----:   |    :----:   |    :----:   |        ---: |
| Test       | 0.8624      | 0.8708      | 0.8660      | 0.8680      |

*The notebook might not load at the first try due to the images weight, please be patient.

## Task2 work
The purpose of this week's task is to learn the techniques for category classification: handcrafted and learned. For this reason, we implement, tune and evaluate a MLP model in order to classify the images and compare its performance with a SVM that uses the net's features perform the classification. In the second part, we need to use the MLP as dense descriptor by training it with image patches, whose predictions are later aggregated to perform the final test. Then, we make use of model features before the classification layer and use them as a dense descriptor and extract the final classification by applying a BoVW algorithm. The best results in the previous experiments are shown next:

| Experiment             |     Accuracy     | 
| :---                   |    :----:        |  
| End-to-end whole image |     0.6072       | 
| Deep features + SVM    |     0.4138       | 
| End-to-end patches     |     0.5328       | 
| Deep features + BoVW   |     **0.6481**       |


## Task3 work
The purpose of this week's task is to learn the techniques for using a already existent CNN with pretrained wheights and fine tunning it. For this reason, we adapt the output and unfreeze some layers so the model can be retrained to our necessities. In the second part, we need to find the best hyper parameter configuration to increase the CNN's performance even further and study what happens when removing different blocks to reduce the CNN's dimensionality.

| Test        |     Dataset     |  Accuracy  |  Loss  |
| :---        |    :----:       |  :------:  | :----: |
| Baseline    |     Large       |  0.9418    | 0.1871 |
| Baseline    |     Small       |  0.9095    | 0.3980 |
| Best model  |     Small       |  0.9542    | 0.1502 |


## Task4 work
For the last week, the objective has been to create a CNN from scratch and obtain the highest accuracy while using the fewer parameters. We have created a baseline model and have performed an exhaustive analysis on how every parameter affected the performance of the baseline model. At the end, we have obtained 2 models were the first one consists on the model with the best ratio (accuracy/(# parameters)/100000) and the second one is the model with the best accuracy. The obtained results are shown next:

| Test             |     Accuracy     |  # Parameters  |  Ratio  |
| :---             |    :----:        |  :------:      | :----:  |
| Baseline         |     0.5610       |  94152         | 0.6002  |
| Best accuracy    |     0.8021       |  41820         | 1.918   |
| Best model       |     0.7645       |  17752         | 0.7645  |
