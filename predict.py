import os
import cv2
from keras.models import load_model
import numpy as np
import logging
import time


# load first model relevant to prediction #
def load_models(input_dir):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"The specified input_dir {input_dir} does not exist")
    dnn_model_path = None
    for file in sorted(os.listdir(input_dir)):
        if file.endswith("h5"):
            dnn_model_path = os.path.join(input_dir, file)
            break
    if dnn_model_path is None:
        raise FileNotFoundError(f"could not find a *.h5 file in {input_dir}")
    load_start_time = time.perf_counter()
    dnn_model = load_model(dnn_model_path)
    logging.info(f"dnn_model: {dnn_model_path} loaded successfully in {time.perf_counter() - load_start_time:.3} seconds")
    return dnn_model


# load image #
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"image {image_path} is None, there was an error while loading")
    image = np.expand_dims(image, axis=0)
    logging.info(f"image {image_path} loaded successfully")
    return image


# perform image class prediction using models and data #
def predict_on_image(image, dnn_model, classes_dict):
    dnn_pred = int(np.argmax(dnn_model.predict(image), axis=1))
    logging.info(f"predicted_class: {classes_dict[dnn_pred]}")
