# DL_project

### Data 

A total of 377,110 chext X-ray images and 227,827 free-text radiology reports were downloaded from this website https://physionet.org/content/mimic-cxr-jpg/2.0.0/. Train, validation and test subsets were created from the full dataset. A total of 10000 images were used for training, 1200 images were used for validation and 1200 images were used for testing. 


### Output

Unimodal and multimodal models saved during training and the output of the model are in the "output" directory. 


### Train

Log files contain the hyperparameters used, train and validation loss and accuracy for each epoch.

#### Unimodal classification using ResNet-50

Pre-trained ResNet-50 was used to train the model with the intermediate layers freezed. 

#### Multimodal classification using CLIP



