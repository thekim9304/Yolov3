import os
import cv2
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET

anchors_wh = np.array([[10, 13], [16, 30], [33, 23],
                       [30, 61], [62, 45], [59, 119],
                       [116, 90], [156, 198], [373, 326]], np.float32) / 416
anchors_wh_mask = np.array([[[10, 13], [16, 30], [33, 23]],
                            [[30, 61], [62, 45], [59, 119]],
                            [[116, 90], [156, 198], [373, 326]]], np.float32) / 416

class Preprocessor:
    def __init__(self, data_dir, output_shape=(416, 416), batch_size=16, labels=None):
        if labels is None:
            raise Exception('Enter the labels, Ex)[raccoon, dog, cat]')
        self.data_dir = data_dir
        self.output_shape = output_shape
        self.labels = labels
        self.num_class = len(labels)
        self.batch_size = batch_size

        files = os.listdir(self.data_dir)
        if 'annotations' in files:
            self.ann_dir = os.path.join(self.data_dir, 'annotations')
        else:
            raise Exception('No exist the \'annotations\' file')
        if 'images' in files:
            self.img_dir = os.path.join(self.data_dir, 'images')
        else:
            raise Exception('No exist the \'images\' file')

        self.all_anns, self.seen_labels = self.parse_annotation(self.ann_dir, self.img_dir)
        self.num_total_data = len(self.all_anns)
        print('Data number : {}, seen labels : {}'.format(self.num_total_data,
                                                          self.seen_labels))

        if self.num_total_data > self.batch_size:
            self.total_batchs = self.num_total_data // self.batch_size

            if (self.num_total_data % self.batch_size) != 0:
                self.total_batchs += 1
        else:
            self.total_batchs = 1

    def __call__(self):
        for i in range(self.total_batchs):
            batch_idx = i
            check_remain_data = (self.num_total_data - (batch_idx * self.batch_size)) // self.batch_size
            if check_remain_data != 0:
                batch_data_num = self.batch_size
            else:
                batch_data_num = self.num_total_data % self.batch_size

            start_idx = (i * self.batch_size)
            end_idx = start_idx + batch_data_num

            imgs = np.empty((1, 416, 416, 3), dtype='float32')
            labels1 = np.empty((1, 52, 52, 3, (4 + 1 + self.num_class)), dtype='float32')
            labels2 = np.empty((1, 26, 26, 3, (4 + 1 + self.num_class)), dtype='float32')
            labels3 = np.empty((1, 13, 13, 3, (4 + 1 + self.num_class)), dtype='float32')

            for ann in self.all_anns[start_idx:end_idx]:
                classes, bboxes = self.parse_y_features(ann)

                label = (
                    self.preprocess_label_for_one_scale(classes, bboxes, 52,
                                                        np.array([0, 1, 2])),
                    self.preprocess_label_for_one_scale(classes, bboxes, 26,
                                                        np.array([3, 4, 5])),
                    self.preprocess_label_for_one_scale(classes, bboxes, 13,
                                                        np.array([6, 7, 8])))

                labels1 = np.append(labels1, np.expand_dims(label[0], axis=0), axis=0)
                labels2 = np.append(labels2, np.expand_dims(label[1], axis=0), axis=0)
                labels3 = np.append(labels3, np.expand_dims(label[2], axis=0), axis=0)

                img = cv2.resize(cv2.imread(ann['filename']).astype('float32') / 255, self.output_shape)
                imgs = np.append(imgs, np.expand_dims(img, axis=0), axis=0)

            imgs = np.delete(imgs, [0, 0], axis=0)
            labels1 = np.delete(labels1, [0, 0], axis=0)
            labels2 = np.delete(labels2, [0, 0], axis=0)
            labels3 = np.delete(labels3, [0, 0], axis=0)
            yield imgs, (labels1, labels2, labels3)

    def preprocess_label_for_one_scale(self, classes, bboxes, grid_size, valid_anchors):
        y = tf.zeros((grid_size, grid_size, 3, 5 + len(self.labels)))

        anchor_indices = self.find_best_anchors(bboxes)

        tf.debugging.Assert(classes.shape[0] == bboxes.shape[0], [classes])
        tf.debugging.Assert(anchor_indices.shape[0] == bboxes.shape[0], [anchor_indices])

        num_boxes = tf.shape(classes)[0]

        indices = tf.TensorArray(tf.int32, 1, dynamic_size=True)
        updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)

        valid_count = 0

        for i in range(num_boxes):
            curr_class = tf.cast(classes[i], tf.float32)
            curr_box = bboxes[i]
            curr_anchor = anchor_indices[i]

            anchor_found = tf.reduce_any(curr_anchor == valid_anchors)
            if anchor_found:
                # ex, anchor index = 7 will have index = 1
                adjusted_anchor_index = tf.math.floormod(curr_anchor, 3)

                # yolo loss를 계산하기 위해서
                # (xmin, ymin, xmax, ymax)를 (centroid x, centroid y, width, height)로 변환
                curr_box_xy = (curr_box[..., 0:2] + curr_box[..., 2:4]) / 2
                curr_box_wh = curr_box[..., 2:4] - curr_box[..., 0:2]

                grid_cell_xy = tf.cast(curr_box_xy // tf.cast((1 / grid_size), dtype=tf.float32), tf.int32)

                index = tf.stack([grid_cell_xy[1], grid_cell_xy[0], adjusted_anchor_index])

                # note that we need to make this one-hot classes in order to use 'categorical crossentropy' later
                update = tf.concat(values=[curr_box_xy, curr_box_wh, tf.constant([1.0]), curr_class], axis=0)

                indices = indices.write(valid_count, index)
                updates = updates.write(valid_count, update)

                valid_anchors += 1

                # y[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                y = tf.tensor_scatter_nd_update(y, indices.stack(), updates.stack())
        return y

    def parse_annotation(self, ann_dir, img_dir):
        all_anns = []
        seen_labels = {}

        for ann in sorted(os.listdir(ann_dir)):
            img = {'object': []}

            tree = ET.parse(os.path.join(ann_dir, ann))

            for elem in tree.iter():
                if 'filename' in elem.tag:
                    img['filename'] = os.path.join(img_dir, elem.text)
                if 'width' in elem.tag:
                    img['width'] = int(elem.text)
                if 'height' in elem.tag:
                    img['height'] = int(elem.text)
                if 'object' in elem.tag or 'part' in elem.tag:
                    obj = {}

                    for attr in list(elem):
                        if 'name' in attr.tag:
                            obj['name'] = attr.text
                            obj['class'] = self.labels.index(obj['name'])

                            if obj['name'] in seen_labels:
                                seen_labels[obj['name']] += 1
                            else:
                                seen_labels[obj['name']] = 1

                            if len(self.labels) > 0 and obj['name'] not in self.labels:
                                break
                            else:
                                img['object'] += [obj]

                        if 'bndbox' in attr.tag:
                            for dim in list(attr):
                                if 'xmin' in dim.tag:
                                    obj['xmin'] = int(round(float(dim.text)))
                                if 'ymin' in dim.tag:
                                    obj['ymin'] = int(round(float(dim.text)))
                                if 'xmax' in dim.tag:
                                    obj['xmax'] = int(round(float(dim.text)))
                                if 'ymax' in dim.tag:
                                    obj['ymax'] = int(round(float(dim.text)))

            if len(img['object']) > 0:
                all_anns += [img]

        all_anns = self.rescale_ann(all_anns)

        return all_anns, seen_labels

    def rescale_ann(self, anns):
        for ann in anns:
            for obj in ann['object']:
                obj['xmin'] = (obj['xmin'] / ann['width'])
                obj['ymin'] = (obj['ymin'] / ann['height'])
                obj['xmax'] = (obj['xmax'] / ann['width'])
                obj['ymax'] = (obj['ymax'] / ann['height'])

        return anns

    def parse_y_features(self, anns):
        classes = []
        bboxes = []
        for obj in anns['object']:
            bboxes.append([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])
            classes.append(obj['class'])

        return tf.one_hot(np.array(classes), len(self.labels)), np.array(bboxes)

    def find_best_anchors(self, bboxes):
        box_wh = bboxes[..., 2:4] - bboxes[..., 0:2]
        box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, tf.shape(anchors_wh)[0], 1))

        intersection = tf.minimum(box_wh[..., 0], anchors_wh[..., 0]) * tf.minimum(box_wh[..., 1], anchors_wh[..., 1])
        box_area = box_wh[..., 0] * box_wh[..., 1]
        anchor_area = anchors_wh[..., 0] * anchors_wh[..., 1]

        iou = intersection / (box_area + anchor_area - intersection)
        anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.int32)

        return anchor_idx

def main():
    preprocess = Preprocessor('../dataset_test', batch_size=2, labels=['raccoon', 'test'])
    num_classes = len(preprocess.labels)

    imgs, labels = next(preprocess())

    print(imgs.shape)
    print(labels[0].shape)
    print(labels[1].shape)
    print(labels[2].shape)

if __name__ == '__main__':
    main()