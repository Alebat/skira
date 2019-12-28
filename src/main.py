import cv2
import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm

from YOLOv3_TensorFlow.model import yolov3
from YOLOv3_TensorFlow.utils.data_aug import letterbox_resize
from YOLOv3_TensorFlow.utils.misc_utils import parse_anchors, read_class_names
from YOLOv3_TensorFlow.utils.nms_utils import gpu_nms
from src.bboxes import crop, highlight
from iou_tracker.iou_tracker import track_iou
from iou_tracker.util import load_mot, save_to_csv
from time import time
import argparse


def main(args):
    time_id = int(time())
    os.makedirs(f'../data/{time_id}', exist_ok=True)

    n = args.name
    detections = []

    ##########
    # YOLOv3 #
    ##########

    args.anchors = parse_anchors(args.anchor_path)
    args.classes = read_class_names(args.class_name_path)
    args.num_class = len(args.classes)

    vid = cv2.VideoCapture(f'../data/vids/{n}.MOV')
    video_frame_count = int(vid.get(7))
    video_fps = int(vid.get(5))

    print(f"Video fps: {video_fps}")

    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
        yolo_model = yolov3(args.num_class, args.anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.4,
                                        nms_thresh=0.45)

        saver = tf.train.Saver()
        saver.restore(sess, args.restore_path)

        start = time()
        for i in tqdm(range(video_frame_count)):
            ret, img_ori = vid.read()
            # img_ori = cv2.rotate(img_ori, cv2.ROTATE_90_CLOCKWISE)
            if args.letterbox_resize:
                img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
            else:
                height_ori, width_ori = img_ori.shape[:2]
                img = cv2.resize(img_ori, tuple(args.new_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img, np.float32)
            img = img[np.newaxis, :] / 255.

            boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

            # rescale the coordinates to the original image
            if args.letterbox_resize:
                boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
                boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
            else:
                boxes_[:, [0, 2]] *= (width_ori / float(args.new_size[0]))
                boxes_[:, [1, 3]] *= (height_ori / float(args.new_size[1]))

            for bi in range(len(boxes_)):
                if args.classes[labels_[bi]] == 'person':
                    x0, y0, x1, y1 = boxes_[bi]
                    detections.append([i, -1, x0, y0, x1, y1, scores_[bi], -1, -1, -1])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    end = time()
    print(f"Detection done in {end - start}s, {(end - start) / video_frame_count}s/f = {video_frame_count / (end - start)}f/s")

    vid.release()

    with open(f'../data/{time_id}/det_{n}.txt', "w") as f:
        for d in detections:
            print(*d, sep=",", file=f)

    ###############
    # iou tracker #
    ###############

    # posso metterci un ndarray in load_mot
    detections = load_mot(np.array(detections))

    start = time()
    tracks = track_iou(detections, args.sigma_l, args.sigma_h, args.sigma_iou, args.t_min)
    end = time()

    num_frames = len(detections)
    print(f"Tracking done in {end - start} at {str(int(num_frames / (end - start)))}fps")

    save_to_csv(f'../data/{time_id}/tracks_{n}.txt', tracks)

    ######################
    # highlight and crop #
    ######################

    highlight(f'../data/vids/{n}.MOV', f'../data/{time_id}/tracks_{n}.txt', f'../data/{time_id}/main_{n}.mov',
              False, False)
    crop(f'../data/vids/{n}.MOV', f'../data/{time_id}/tracks_{n}.txt', f'../data/{time_id}/crop_{n}', 'mov',
         False, False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="IOU Tracker demo script")
    # parser.add_argument('-d', '--detection_path', type=str, required=True,
    #                     help="full path to CSV file containing the detections")
    # parser.add_argument('-o', '--output_path', type=str, required=True,
    #                     help="output path to store the tracking results (MOT challenge devkit compatible format)")
    parser.add_argument('-sl', '--sigma_l', type=float, default=0,
                        help="low detection threshold")
    parser.add_argument('-sh', '--sigma_h', type=float, default=0.5,
                        help="high detection threshold")
    parser.add_argument('-si', '--sigma_iou', type=float, default=0.5,
                        help="intersection-over-union threshold")
    parser.add_argument('-tm', '--t_min', type=float, default=60,
                        help="minimum track length")
    parser.add_argument("name", type=str,
                        help="The name of the input video.")
    parser.add_argument("--anchor_path", type=str, default="../YOLOv3_TensorFlow/data/yolo_anchors.txt",
                        help="The path of the anchor txt file.")
    parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                        help="Resize the input image with `new_size`, size format: [width, height]")
    parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Whether to use the letterbox resize.")
    parser.add_argument("--class_name_path", type=str, default="../YOLOv3_TensorFlow/data/coco.names",
                        help="The path of the class names.")
    parser.add_argument("--restore_path", type=str, default="../YOLOv3_TensorFlow/data/darknet_weights/yolov3.ckpt",
                        help="The path of the weights to restore.")

    args = parser.parse_args()
    main(args)
