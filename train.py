import os
import datetime

import tensorflow as tf

from model.yolov3 import YoloV3, YoloLoss
from utils.preprocess import Preprocessor, anchors_wh_mask

BATCH_SIZE = 8
EPOCH = 10


def main():
    data_dir = './dataset_test'
    labels = ['raccoon']
    num_classes = len(labels)

    ckpt_dir = './checkpoints'

    lr_rate = 0.0001

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    preprocessor = Preprocessor(data_dir=data_dir,
                                output_shape=(416, 416),
                                labels=labels,
                                batch_size=BATCH_SIZE)

    model = YoloV3(input_shape=(416, 416, 3), num_classes=num_classes, training=True)
    loss_objects = [YoloLoss(valid_anchors_wh, num_classes) for valid_anchors_wh in anchors_wh_mask]

    optimizer = tf.keras.optimizers.Adam(lr=lr_rate)

    for epoch in range(EPOCH):
        print('{} epoch start! : {}'.format(epoch, datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")))

        train_one_epoch(model, loss_objects, preprocessor(), optimizer)


def train_one_epoch(model, loss_objects, generator, optimizer):
    epoch_total_loss = 0.0
    epoch_xy_loss = 0.0
    epoch_wh_loss = 0.0
    epoch_class_loss = 0.0
    epoch_obj_loss = 0.0
    batchs = 0
    for i, (images, labels) in enumerate(generator):
        batch_size = images.shape[0]
        with tf.GradientTape() as tape:
            outputs = model(images)

            total_losses, xy_losses, wh_losses, class_losses, obj_losses = [], [], [], [], []

            # iterate over all three sclaes
            for loss_object, y_pred, y_true in zip(loss_objects, outputs, labels):
                total_loss, loss_breakdown = loss_object(y_true, y_pred)
                xy_loss, wh_loss, class_loss, obj_loss = loss_breakdown
                total_losses.append(total_loss * (1. / batch_size))
                xy_losses.append(xy_loss * (1. / batch_size))
                wh_losses.append(wh_loss * (1. / batch_size))
                class_losses.append(class_loss * (1. / batch_size))
                obj_losses.append(obj_loss * (1. / batch_size))

            total_loss = tf.reduce_sum(total_losses)
            total_xy_loss = tf.reduce_sum(xy_losses)
            total_wh_loss = tf.reduce_sum(wh_losses)
            total_class_loss = tf.reduce_sum(class_losses)
            total_obj_loss = tf.reduce_sum(obj_losses)

        grads = tape.gradient(target=total_loss, sources=model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_total_loss += total_loss
        epoch_xy_loss += total_xy_loss
        epoch_wh_loss += total_wh_loss
        epoch_class_loss += total_class_loss
        epoch_obj_loss += total_obj_loss
        batchs += 1

    epoch_total_loss = epoch_total_loss / batchs
    print(' total_loss.[{0:.4f}]'.format(epoch_total_loss))
    print('     xy:{:.4f}, wh:{:.4f}, class:{:.4f}, obj:{:.4f}'.format(epoch_xy_loss / batchs,
                                                                       epoch_wh_loss / batchs,
                                                                       epoch_class_loss / batchs,
                                                                       epoch_obj_loss / batchs))


if __name__ == '__main__':
    main()
