import csv
import cv2
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, matthews_corrcoef


# load dataset using csv file, data_dir and classes dictionary
def load_dataset(csv_file, data_dir, cls_dict):
    logging.info("Loading images and creating train and test sets...", flush=True)
    image_label_dict = {}  # create a dictionary mapping image name to label
    with open(csv_file, 'r') as csv_file_f:  # read csv file
        csv_file_reader = csv.DictReader(csv_file_f)
        for row in csv_file_reader:
            image_label_dict[row["IMAGE_FILENAME"].strip()] = row["LABEL"].strip()

    num_classes = len(cls_dict)
    image_names = sorted(os.listdir(data_dir))
    for image_num, image_name in enumerate(tqdm(image_names)):  # read images
        image_path = os.path.join(data_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("image %s is None, there was an error while loading" % image_path)
        image = np.expand_dims(image, axis=0)

        # initialize dataset arrays #
        if image_num == 0:
            H, W, C = np.shape(image)[1:]
            dataset_images = np.zeros((0, H, W, C))
            dataset_labels = np.zeros((0, num_classes))

        # process label - one-hot encoding #
        label = np.zeros((1, num_classes))
        label[:, cls_dict[image_label_dict[image_name]]] = 1

        # aggregate to dataset #
        dataset_images = np.concatenate((dataset_images, image), axis=0)
        dataset_labels = np.concatenate((dataset_labels, label), axis=0)
    return dataset_images, dataset_labels


# get weighting to mitigate class imbalance #
def get_class_weighting(labels, num_classes_for_weighting):
    sample_scalar = 0
    sample_weights = np.zeros((len(labels)))
    for ii in range(num_classes_for_weighting):
        sample_scalar += 1 / np.sum(labels[:, ii])
    for ii in range(num_classes_for_weighting):
        sample_weights[np.where(labels[:, ii] == 1)[0]] = (1 / np.sum(labels[:, ii])) / sample_scalar
    return sample_weights


# gather key performance metrics #
def get_model_performance(model_to_test, x_to_test, y_to_test, sample_weighting):
    y_pred = np.argmax(model_to_test.predict(x_to_test), axis=1)
    y_true = np.argmax(y_to_test, axis=1)

    results = dict(
            f1_score_micro=f1_score(y_true=y_true, y_pred=y_pred, average="micro"),
            f1_score_macro=f1_score(y_true=y_true, y_pred=y_pred, average="macro"),
            f1_score_weighted=f1_score(y_true=y_true, y_pred=y_pred, average="weighted"),
            weighted_accuracy=accuracy_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weighting),
            unweighted_accuracy=accuracy_score(y_true=y_true, y_pred=y_pred),
            matthews_corrcoef_weighted=matthews_corrcoef(y_true=y_true, y_pred=y_pred),
            classifier_confusion_matrix=confusion_matrix(y_true=y_true, y_pred=y_pred)
    )
    return results
