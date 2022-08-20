import os
import predict
import train
import logging
import argparse
from constants import *


# main run function
def main(args):
    """
    :param args - inputs from user
    :return:
    """
    # start logging
    logging.basicConfig(level=logging.DEBUG)

    # prediction mode
    if args.mode == "predict":
        if not os.path.isfile(args.image_path):
            raise ValueError(f"the specified image path {args.image_path} is not valid")

        # load model
        dnn_model = predict.load_local_model(input_dir="saved_model")
        # load image
        image = predict.load_image(image_path=args.image_path)
        # predict image class
        logging.info(f"Performing inference on {args.image_path}")
        predict.predict_on_image(image=image,
                                 dnn_model=dnn_model,
                                 classes_dict=CLASSES_DICT)
    # training mode
    elif args.mode == "train":
        if not args.baseline_dir:
            raise ValueError("path of dataset directory for training isn't specified")

        # log hyper-parameters
        logging.info(f"Training hyper-parameters: lr:{LEARNING_RATE}, batch_size:{BATCH_SIZE}, NUM_EPOCHS:{NUM_EPOCHS}, TEST_RATIO:{TEST_RATIO}")
        # train the DNN model
        train.main(test_ratio=TEST_RATIO,
                   learning_rate=LEARNING_RATE,
                   num_classes=NUM_CLASSES,
                   num_epochs=NUM_EPOCHS,
                   mini_batch_size=BATCH_SIZE,
                   classes_dict=CLASSES_DICT,
                   baseline_dir=args.baseline_dir)

    # illegal mode
    else:
        raise ValueError(f"invalid option {args.mode}")


# main part, used for either training the model, or predicting the classification of an input image #
if __name__ == "__main__":
    # Create parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["predict", "train"], required=True, help="determine whether to train the model or perform prediction")
    parser.add_argument("--image_path", type=str, help="path of image to perform prediction on")
    parser.add_argument("--baseline_dir", type=str, help="path of directory from which to fetch data for training")
    args = parser.parse_args()

    # call main function with parsed arguments
    main(args=args)
