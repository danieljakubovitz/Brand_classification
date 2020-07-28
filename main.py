import sys
import predict
import train


# main run function
def main(*args):
    args = args[0]
    mode = args[1]
    # prediction mode
    if mode in ["predict", "-predict"]:
        if len(args) < 3:
            raise SyntaxError("path of image to predict isn't specified.")

        # load models
        dnn_model = predict.load_models(input_dir="saved_model")
        # load image and its metadata
        image = predict.load_image(image_path=args[2])
        # predict image class
        predict.predict_on_image(image=image,
                                 dnn_model=dnn_model)
    # training mode
    elif mode in ["train", "-train"]:
        if len(args) < 3:
            raise SyntaxError("path of dataset directory for training isn't specified.")
        train.main(test_ratio=0.1,
                   initial_learning_rate=5e-5,
                   num_classes=3,
                   num_training_epochs=15,
                   classes_dict={"CLASS_0": 0,
                                 "CLASS_1": 1,
                                 "CLASS_2": 2},
                   baseline_dir=args[2])

    # illegal mode
    else:
        raise Exception("invalid option %s" % args[1])


# main part, used for either training the model, or predicting the classification of an input image #
if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SyntaxError("Insufficient arguments.")
    main(sys.argv)
