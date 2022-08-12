import os
import predict
import train
import logging
import argparse

# main run function
def main(args):
    # start logging
    logging.basicConfig(level=logging.DEBUG)

    # prediction mode
    if args.mode in ["predict", "-predict"]:
        if not os.path.isfile(args.image_path):
            raise ValueError(f"specified image path {args.image_path} is not valid")

        # load models
        dnn_model = predict.load_models(input_dir="saved_model")
        # load image
        image = predict.load_image(image_path=args.image_path)
        # predict image class
        predict.predict_on_image(image=image,
                                 dnn_model=dnn_model,
                                 classes_dict={0: "CLASS_0",
                                               1: "CLASS_1",
                                               2: "CLASS_2"}
                                 )
    # training mode
    elif args.mode in ["train"]:
        if not args.baseline_dir:
            raise SyntaxError("path of dataset directory for training isn't specified")

        train.main(test_ratio=0.1,
                   initial_learning_rate=5e-5,
                   num_classes=3,
                   num_training_epochs=15,
                   classes_dict={"CLASS_0": 0,
                                 "CLASS_1": 1,
                                 "CLASS_2": 2},
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
    main(args)
