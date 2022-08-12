import os
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense, Input, Flatten
from sklearn.model_selection import train_test_split
import time
from utils import load_dataset, get_class_weighting, get_model_performance
import logging


# main training function #
def main(test_ratio, initial_learning_rate, num_classes, num_training_epochs, classes_dict, baseline_dir):
    start_time = time.perf_counter()  # run-time evaluation

    # csv file full path, and images dir full puth
    csv_file = None
    data_dir = None
    for file in os.listdir(baseline_dir):
        if file.endswith("csv"):
            csv_file = os.path.join(baseline_dir, file)
        if os.path.isdir(os.path.join(baseline_dir, file)):
            data_dir = os.path.join(baseline_dir, file)
    if csv_file is None:
        raise FileNotFoundError("no csv_file was found")
    if data_dir is None:
        raise FileNotFoundError("no data_dir was found")

    # get all relevant data #
    dataset_images, dataset_labels = load_dataset(csv_file=csv_file,
                                                  data_dir=data_dir,
                                                  cls_dict=classes_dict)

    # get input image shape #
    H, W = np.shape(dataset_images)[1:3]

    # split to data to train and test, while keeping class ratio in both train and test #
    X_train, X_test, Y_train, Y_test = train_test_split(dataset_images, dataset_labels,
                                                        stratify=dataset_labels,
                                                        test_size=test_ratio,
                                                        random_state=42,
                                                        shuffle=True)

    logging.info(f"Training set size: {len(X_train)} samples")
    logging.info(f"Test set size: {len(X_test)} samples")

    # TRAIN A DNN FOR 3 CLASS CLASSIFICATION #
    dnn_model = keras.Sequential(name="full_DNN")
    dnn_model_input = Input(shape=(H,W,3), name="dnn_model_input")

    # use VGG16 model with ImageNet weights as baseline #
    init_dnn_model = tf.keras.applications.VGG19(
                                include_top=False, # do not include 3 FC layers at top of network
                                input_tensor=dnn_model_input,
                                weights="imagenet",
                                input_shape=(H, W, 3),
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    dnn_model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=[tf.keras.metrics.CategoricalAccuracy(),
                               tf.keras.metrics.AUC(curve="PR")])

    # get train and test sample weighting, according to class imbalance
    train_sample_weights = 100 * get_class_weighting(labels=Y_train,
                                                     num_classes_for_weighting=num_classes)
    test_sample_weights = get_class_weighting(labels=Y_test,
                                              num_classes_for_weighting=num_classes)

    logging.info("Starting dnn_model training...")
    dnn_model.fit(
        x=X_train,
        y=Y_train,
        batch_size=32,
        epochs=num_training_epochs,
        verbose=1,
        validation_split=0.0,
        validation_data=(X_test, Y_test),
        shuffle=True,
        sample_weight=train_sample_weights,
    )

    # save trained DNN model #
    dest_dir = "saved_model"
    os.makedirs(dest_dir, exist_ok=True)
    logging.info("Saving trained DNN model to \"saved_model\" directory..")
    dnn_model.save(os.path.join(dest_dir, "dnn_model.h5"))
    logging.info(f"Model saved to: {os.path.join(dest_dir, 'dnn_model.h5')}")


    # evaluate classifier #
    model_stats = get_model_performance(model_to_test=dnn_model,
                                        x_to_test=X_test,
                                        y_to_test=Y_test,
                                        sample_weighting=test_sample_weights)

    logging.info("DNN model TEST performance:")
    for key, val in model_stats.items():
        logging.info(f"{key}: {val}")

    logging.info("total run time: %g minutes" % ((time.perf_counter()-start_time)/60))
