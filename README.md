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
| Deep features + BoVW   |     0.6481       |


## Task3 work

## Task4 work

