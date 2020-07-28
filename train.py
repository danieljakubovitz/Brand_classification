# first, import some necessary libraries #
import os
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Concatenate, Dense, Input, Flatten
from sklearn.model_selection import train_test_split
import time
from utils import load_dataset, get_class_weighting, get_model_performance


# main training function #
def main(test_ratio, initial_learning_rate, num_classes, num_training_epochs, classes_dict, baseline_dir):
    start_time = time.time()  # run-time evaluation

    # csv file full path, and images dir full puth
    csv_file = os.path.join(baseline_dir, "gicsd_labels.csv")
    data_dir = os.path.join(baseline_dir, "images")

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

    # TRAIN A DNN FOR 3 CLASS CLASSIFICATION #
    DNN_model = keras.Sequential(name="full_DNN")
    DNN_model_input = Input(shape=(H,W,1), name="DNN_model_input")
    # input is concatenated 3 times along channel axis #
    concat_input = Concatenate(axis=3)([DNN_model_input, DNN_model_input, DNN_model_input])

    # use VGG16 model with imagenet weights as baseline #
    init_DNN_model = keras.applications.VGG16(
                                include_top=False,
                                input_tensor=concat_input,
                                weights="imagenet",
                                input_shape=(H, W, 3),
                            )

    # add fully connected layers in the end of the network #
    DNN_model.add(init_DNN_model)
    DNN_model.add(Flatten(name="flatten1"))  # make sure output is flat before FC layers
    DNN_model.add(Dense(units=1024, activation="relu", name="FC1",
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-8, l2=1e-8),
                    bias_regularizer=tf.keras.regularizers.l2(1e-8),
                    ))
    DNN_model.add(Dense(256, activation="relu", name="FC2",
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-8, l2=1e-8),
                    bias_regularizer=tf.keras.regularizers.l2(1e-8),
                    ))
    DNN_model.add(Dense(num_classes, activation="softmax", name="FC3",
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-8, l2=1e-8),
                    bias_regularizer=tf.keras.regularizers.l2(1e-8),
                    ))

    # user Adam optmizer for speedy learning #
    optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
    DNN_model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])

    # get train and test sample weighting, according to class imbalance
    train_sample_weights = 100 * get_class_weighting(labels=Y_train,
                                                     num_classes_for_weighting=num_classes)
    test_sample_weights = get_class_weighting(labels=Y_test,
                                              num_classes_for_weighting=num_classes)

    print("Starting DNN_model training...")
    DNN_model.fit(
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
    dest_dir = "artifacts"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    print("Saving trained DNN model to artifacts directory..")
    DNN_model.save(os.path.join(dest_dir, "DNN_model.h5"))

    # evaluate classifier #
    model_stats = get_model_performance(model_to_test=DNN_model,
                                        x_to_test=X_test,
                                        y_to_test=Y_test,
                                        sample_weighting=test_sample_weights)

    print("DNN model performance:")
    for key, val in model_stats.items():
        print("{}: {}".format(key, val))

    print("total run time: %g minutes" % ((time.time()-start_time)/60))
