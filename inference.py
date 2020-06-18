import cv2
import tensorflow as tf

from utils.postprocess import Postprocessor
from model.yolov3 import YoloV3


def main():
    # Setting
    input_shape = (413, 413, 3)
    classes = ['raccoon']
    num_classes = len(classes)

    # Load the trained model
    model = YoloV3(input_shape=input_shape, num_classes=num_classes, training=False)
    ckpt_path = './checkpoints_test'
    ckpt = tf.train.latest_checkpoint(ckpt_path)
    model.load_weights(ckpt)

    # Prepare a input image
    img = cv2.imread('./dataset_test/images/raccoon-1.jpg')
    x = tf.expand_dims(cv2.resize(img.astype('float32') / 255, input_shape[:2]), axis=0)

    # Postprocessor object
    postprocessor = Postprocessor(0.5, 0.5, 2)

    # Predict
    y_pred = model(x)

    # Postprocess
    boxes, scores, classes, num_detection = postprocessor(y_pred)

    num_img = num_detection.shape[0]
    for img_i in range(num_img):
        # based on train data
        # h, w, d = x[img_i].shape
        # based on original image
        h, w, d = img.shape

        for i in range(num_detection[img_i][0].numpy()):
            box = boxes[img_i][i].numpy()

            xmin = int(box[0] * w)
            ymin = int(box[1] * h)
            xmax = int(box[2] * w)
            ymax = int(box[3] * h)

            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    cv2.imshow('t', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
