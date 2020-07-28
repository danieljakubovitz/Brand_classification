This repo includes a generic 3-class classification model (easily extendable to more classes).
It includes the training of a DNN, using transfer learning from VGG-16 initialized with ImageNet.
Sample loss weighting is incorporated to mitigate class imbalance.

This repo is based on keras + tensorflow backend.

All training and test data should be aggregated in a directory, along with a csv file describing its contents using 2 fields:
1. IMAGE_FILENAME
2. LABEL
The data directory should include the csv file and an images directory with the images themselves.

The dictionary classes_dict in the main file should match the classes in the dataset's csv file.

Training:
python main.py -train path_to_data_directory

The saved model will appear in a "saved_model" directory.

Inference:
python main.py -predict path_of_image_to_predict