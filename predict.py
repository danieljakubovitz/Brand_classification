from utils import transform_to_one_channel
import os
import cv2
from keras.models import load_model
import numpy as np

classes_dict = {0: "CLASS_0",
                1: "CLASS_1",
                2: "CLASS_2"}

# load all models relevant to prediction #
def load_models(input_dir):
    DNN_model_path = os.path.join(input_dir, "DNN_model.h5")
    if not os.path.exists(DNN_model_path):
        raise Exception("DNN model doesn't exists in artifacts directory")
    DNN_model = load_model(DNN_model_path)
    print("DNN_model loaded successfully")
    return DNN_model


# load image and metadata #
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise Exception("image %s is None, there was an error while loading" % image_path)

    card_id = np.expand_dims(int(image_path.split("_")[1]), axis=(0,1))
    background_id = np.expand_dims(int(image_path.split("_")[2]), axis=(0,1))
    image_id = np.expand_dims(int(image_path.split("_")[3].split(".")[0]), axis=(0,1))
    print("image %s loaded successfully" % image_path)
    return image, card_id, background_id, image_id


# perform image class prediction using models and data #
def predict_on_image(image, DNN_model):
    image = transform_to_one_channel(image)
    dnn_pred = int(np.argmax(DNN_model.predict(image), axis=1))
    print("predicted_class: %s" % classes_dict[dnn_pred])
