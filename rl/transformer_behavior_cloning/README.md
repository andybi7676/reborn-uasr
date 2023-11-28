# Code for Behavior Cloning Using Transformers

- The current code hard-code the data path in the `TokenClassificationEmbeddingsDataset`. You need to change each file path in the dataset to run.
- The current code is for predicting the k-means cluster. Change the label to \*.boundaries for boundary prediction. You also need to change the `num_labels = 128`.
