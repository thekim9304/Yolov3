import tensorflow as tf

# from model.yolov3 import YoloV3
from utils.utils import broadcast_iou, xywh_to_x1x2y1y2
from utils.preprocess import Preprocessor

class Postprocessor:
    def __init__(self, iou_threshold=None, score_threshold=None, max_detection=100):
        if iou_threshold == None or score_threshold == None:
            raise Exception('Set the iou_thresh and the score_thresh')

        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_detection = max_detection

    def __call__(self, raw_yolo_outputs):
        boxes, objectness, class_probs = [], [], []

        for raw_yolo_out in raw_yolo_outputs:
            batch_size = tf.shape(raw_yolo_out[0])[0]
            num_classes = tf.shape(raw_yolo_out[2])[-1]
            # need to translate from xywh to x1y1x2y2 format
            boxes.append(tf.reshape(raw_yolo_out[0], (batch_size, -1, 4)))
            objectness.append(tf.reshape(raw_yolo_out[1], (batch_size, -1, 1)))
            class_probs.append(tf.reshape(raw_yolo_out[2], (batch_size, -1, num_classes)))

        boxes = xywh_to_x1x2y1y2(tf.concat(boxes, axis=1))
        objectness = tf.concat(objectness, axis=1)
        class_probs = tf.concat(class_probs, axis=1)

        final_boxes, final_scores, final_classes, valid_detection = self.batch_non_maximum_suppression(
            boxes, objectness, class_probs)

        return final_boxes, final_scores, final_classes, valid_detection

    def batch_non_maximum_suppression(self, boxes, scores, classes):
        def single_batch_nms(candidate_boxes):
            # filter out predictions with score less than score_threshold
            candidate_boxes = tf.boolean_mask(
                candidate_boxes, candidate_boxes[..., 4] >= self.score_threshold)
            outputs = tf.zeros((self.max_detection + 1,
                                tf.shape(candidate_boxes)[-1]))
            indices = []
            updates = []

            count = 0
            # keep running this until there's no more candidate box or max_detection is met
            while tf.shape(candidate_boxes)[0] > 0 and count < self.max_detection:
                # pick the box with the highest score
                best_idx = tf.math.argmax(candidate_boxes[..., 4], axis=0)
                best_box = candidate_boxes[best_idx]
                # add this best box to the output
                indices.append([count])
                updates.append(best_box)
                count += 1
                # remove this box from candidate boxes
                candidate_boxes = tf.concat([
                    candidate_boxes[0:best_idx],
                    candidate_boxes[best_idx + 1:tf.shape(candidate_boxes)[0]]
                ],
                                            axis=0)
                # calculate IOU between this box and all remaining candidate boxes
                iou = broadcast_iou(best_box[0:4], candidate_boxes[..., 0:4])
                # remove all candidate boxes with IOU bigger than iou_threshold
                candidate_boxes = tf.boolean_mask(candidate_boxes,
                                                  iou[0] <= self.iou_threshold)
            if count > 0:
                # also append num_detection to the result
                count_index = [[self.max_detection]]
                count_updates = [
                    tf.fill([tf.shape(candidate_boxes)[-1]], count)
                ]
                indices = tf.concat([indices, count_index], axis=0)
                updates = tf.concat([updates, count_updates], axis=0)
                outputs = tf.tensor_scatter_nd_update(outputs, indices,
                                                      updates)
            return outputs

        combined_boxes = tf.concat([boxes, scores, classes], axis=2)
        result = tf.map_fn(single_batch_nms, combined_boxes)
        # take out num_detection from the result
        valid_counts = tf.expand_dims(
            tf.map_fn(lambda x: x[self.max_detection][0], result), axis=-1)
        final_result = tf.map_fn(lambda x: x[0:self.max_detection], result)
        nms_boxes, nms_scores, nms_classes = tf.split(
            final_result, [4, 1, -1], axis=-1)
        return nms_boxes, nms_scores, nms_classes, tf.cast(
            valid_counts, tf.int32)

# def main():
#     preprocess = Preprocessor('../dataset_test', batch_size=2, labels=['raccoon', 'test'])
#     num_classes = len(preprocess.labels)
#
#     imgs, labels = next(preprocess())
#
#     yolov3 = YoloV3(input_shape=(416, 416, 3), num_classes=num_classes, training=False)
#     outputs = yolov3(imgs)
#
#     postprocess = Postprocessor(0.5, 0.5, 3)
#     boxes, scores, classes, num_detection = postprocess(outputs)
#
#     print(boxes)
#     print(scores)
#     print(classes)
#     print(num_detection)
#
# if __name__ == '__main__':
#     main()