## General
This repo includes a generic 3-class classification model (easily extendable to more classes).
It includes the training of a DNN, using transfer learning from EfficientNetB3 initialized with ImageNet.
Sample loss weighting is incorporated to mitigate class imbalance during training.
Hyper-parameter tuning is necessary to optimize performance.

This repo is based on keras 2.6.0 + tensorflow backend 2.6.2.

All training and test data should be aggregated in a directory, along with a csv file describing its contents using 2 fields:
1. IMAGE_FILENAME (e.g. image.jpg)
2. LABEL (class name, e.g. CLASS_0)

The data directory should include the csv file and an "images" directory with the images themselves.
The dictionary INT_TO_STR_CLASSES_DICT in the constants.py file should match the classes in the dataset csv file.

### Training:
command line example:

python main.py --mode train --baseline_dir "data"

The saved model will appear in a "saved_model" directory.

### Inference:
command line example:

python main.py --mode predict --image_path "data/images/image_0.jpg"

### Evaluation metrics:
The trained model is evaluated using the following metrics:
1. micro F1 score: representing the global F1 score of all the samples.
2. macro F1 score: the unweighted average of the different classes' F1 scores.
3. weighted F1 score: the weighted average (according to the support) of the different classes' F1 scores.
4. weighted accuracy: model accuracy, while accounting for the class imbalance using weights (emphasizing the less common classes more than their respective supports).
5. unweighted accuracy: model accuracy, without accounting for the class imbalance.
6. Matthew's correlation coefficient: a coefficient between -1 and +1, with +1 representing perfect prediction. This metric also accounts for a classifier's true negatives.
7. confusion matrix: this allows us to see absolute numbers in the classification (rather than just ratios), and to better understand the nature of our model's mistakes.
