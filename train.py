import os
import numpy as np
import logging
import time
from sklearn.model_selection import train_test_split
from utils import load_dataset, get_class_weighting, get_sample_weights, get_model_performance, build_model


# main training function #
def main(test_ratio, learning_rate, num_classes, num_epochs, mini_batch_size, classes_dict, baseline_dir):
    start_time = time.perf_counter()  # run-time evaluation

    # csv file full path, and images dir full path
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
    # track class imbalance
    train_class_split = [sum(Y_train[:, ii]) / np.sum(Y_train, axis=(0,1)) for ii in range(num_classes)]
    test_class_split = [sum(Y_test[:, ii]) / np.sum(Y_test, axis=(0,1)) for ii in range(num_classes)]

    logging.info(f"Training set size: {len(X_train)} samples")
    logging.info(f"Training set class split: {train_class_split}")
    logging.info(f"Test set size: {len(X_test)} samples")
    logging.info(f"Test set class split: {test_class_split}")

    # get train and test sample weighting, according to class imbalance
    train_class_weights = get_class_weighting(class_split=train_class_split)
    logging.info(f"Training with class weighting: {train_class_weights}")
    train_sample_weights = get_sample_weights(class_weights=train_class_weights, y=Y_train)

    test_class_weights = get_class_weighting(class_split=test_class_split)
    logging.info(f"Testing with class weighting: {test_class_weights}")
    test_sample_weights = get_sample_weights(class_weights=test_class_weights, y=Y_test)

    # TRAIN A DNN FOR N-CLASS CLASSIFICATION #
    logging.info("Starting dnn_model training...")
    dnn_model = build_model(input_shape=(H,W,3),
                            learning_rate=learning_rate,
                            num_classes=num_classes)
    dnn_model.fit(
        x=X_train,
        y=Y_train,
        batch_size=mini_batch_size,
        epochs=num_epochs,
        verbose=1,
        validation_split=0.0,
        validation_data=(X_test, Y_test, test_sample_weights),
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
