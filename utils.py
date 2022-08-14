import csv
import cv2
import numpy as np
import os
import logging
import tensorflow as tf
import keras
from keras.layers import Dense, Input, Flatten
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, matthews_corrcoef
from tqdm import tqdm

# load dataset using csv file, data_dir and classes dictionary
def load_dataset(csv_file, data_dir, cls_dict):
    logging.info("Loading images and creating train and test sets...")
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
            raise FileNotFoundError(f"image {image_path} is None, there was an error while loading")
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


# get normalized inverse weighting to mitigate class imbalance #
def get_class_weighting(class_split):
    class_weights = [1/x for x in class_split]
    normalized_class_weights = [x/sum(class_weights) for x in class_weights]
    return normalized_class_weights

# get weighting vector matching sample vector dimensions
def get_sample_weights(class_weights, y):
    y_integers = np.argmax(y, axis=1)
    sample_weights = np.array(y_integers).copy()
    for class_idx, weight in enumerate(class_weights):
        indices = np.argwhere(np.array(class_idx) == y_integers).flatten()
        sample_weights[indices] = weight
    return sample_weights



# gather key performance metrics #
def get_model_performance(model_to_test, x_to_test, y_to_test, sample_weighting):
    y_pred = np.argmax(model_to_test.predict(x_to_test), axis=1)
    y_true = np.argmax(y_to_test, axis=1)

    return dict(
            f1_score_micro=f1_score(y_true=y_true, y_pred=y_pred, average="micro"),
            f1_score_macro=f1_score(y_true=y_true, y_pred=y_pred, average="macro"),
            f1_score_weighted=f1_score(y_true=y_true, y_pred=y_pred, average="weighted"),
            weighted_accuracy=accuracy_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weighting),
            unweighted_accuracy=accuracy_score(y_true=y_true, y_pred=y_pred),
            matthews_corrcoef_weighted=matthews_corrcoef(y_true=y_true, y_pred=y_pred),
            classifier_confusion_matrix=confusion_matrix(y_true=y_true, y_pred=y_pred)
    )

# build DNN model #
def build_model(input_shape, learning_rate, num_classes):
    dnn_model = keras.Sequential(name="full_DNN")
    dnn_model_input = Input(shape=input_shape, name="dnn_model_input")

    # use VGG16 model with ImageNet weights as baseline #
    init_dnn_model = tf.keras.applications.VGG19(
                                include_top=False, # do not include 3 FC layers at top of network
                                input_tensor=dnn_model_input,
                                weights="imagenet",
                                input_shape=input_shape,
                            )

    # add fully connected layers in the end of the network #
    dnn_model.add(init_dnn_model)
    dnn_model.add(Flatten(name="flatten1"))  # make sure output is flat before FC layers
    dnn_model.add(Dense(units=1024, activation="relu", name="FC1",
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-8, l2=1e-8),
                        bias_regularizer=tf.keras.regularizers.l2(1e-8),
                        ))
    dnn_model.add(Dense(units=256, activation="relu", name="FC2",
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-8, l2=1e-8),
                        bias_regularizer=tf.keras.regularizers.l2(1e-8),
                        ))
    dnn_model.add(Dense(units=num_classes, activation="softmax", name="FC3",
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-8, l2=1e-8),
                        bias_regularizer=tf.keras.regularizers.l2(1e-8),
                    ))

    # user Adam optimizer for speedy learning #
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    dnn_model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=[tf.keras.metrics.CategoricalAccuracy(),
                               tf.keras.metrics.AUC(curve="PR")])

    return dnn_model