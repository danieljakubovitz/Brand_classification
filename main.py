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
        DNN_model = predict.load_models(input_dir="artifacts")
        # load image and its metadata
        image, card_id, background_id, image_id = predict.load_image(image_path=args[2])
        # predict image class
        predict.predict_on_image(image=image,
                                 DNN_model=DNN_model)
    # training mode
    elif mode in ["train", "-train"]:
        train.main(test_ratio=0.1,
                   initial_learning_rate=1e-4,
                   num_classes=3,
                   num_training_epochs=10,
                   classes_dict={"BRAND_0": 0,
                                 "BRAND_1": 1,
                                 "BRAND_2": 2},
                   baseline_dir="data")

    # illegal mode
    else:
        raise Exception("invalid option %s" % args[1])


# main part, used for either training the model, or predicting the classification of an input image #
if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SyntaxError("Insufficient arguments.")
    main(sys.argv)
