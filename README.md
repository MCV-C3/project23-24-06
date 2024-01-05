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

## Task3 work

## Task4 work

