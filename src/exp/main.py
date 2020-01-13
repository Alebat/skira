from time import time

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from YOLOv3_TensorFlow.model import yolov3
from YOLOv3_TensorFlow.utils.data_aug import letterbox_resize as letterbox_resize_f
from YOLOv3_TensorFlow.utils.misc_utils import parse_anchors, read_class_names
from YOLOv3_TensorFlow.utils.nms_utils import gpu_nms
from iou_tracker.iou_tracker import track_iou
from iou_tracker.util import load_mot, save_to_csv


def iou_tracking(args, time_id, detections):
    print(f'Doing iou_tracking on "{args.video}" ({time_id})')
    # posso metterci un ndarray in load_mot
    detections = load_mot(np.array(detections))
    start = time()
    tracks = track_iou(detections, args.sigma_l, args.sigma_h, args.sigma_iou, args.t_min, args.ttl, args.mom_alpha, args.exp_zoom)
    end = time()
    num_frames = len(detections)
    print(f"Tracking done in {end - start} at {str(int(num_frames / (end - start)))}fps")
    save_to_csv(f'../data/{time_id}/tracks_{args.name}.txt', tracks)


def iou_mom_tracking(detections, sigma_l, sigma_h, sigma_iou, t_min, ttl, mom_alpha, exp_zoom, fps, min_area=0):
    detections = load_mot(np.array(detections))
    return track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min*fps, ttl*fps, mom_alpha, exp_zoom ** (fps / 60),
                     min_area)


def people_detection(video, class_name_path, anchor_path, new_size, restore_path,
                     letterbox_resize):
    anchors = parse_anchors(anchor_path)
    classes = read_class_names(class_name_path)
    num_class = len(classes)
    if isinstance(video, str):
        video = cv2.VideoCapture(video)
    video_frame_count = int(video.get(7))
    video_fps = int(video.get(5))

    print(f'Doing detection on "{video}"')
    print(f"Video fps: {video_fps}")

    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, new_size[1], new_size[0], 3], name='input_data')
        yolo_model = yolov3(num_class, anchors)
        with tf.variable_scope('yolov3', reuse=tf.AUTO_REUSE):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_class, max_boxes=200, score_thresh=0.4,
                                        nms_thresh=0.45)

        saver = tf.train.Saver()
        saver.restore(sess, restore_path)

        start = time()
        for i in tqdm(range(video_frame_count)):
            ret, img_ori = video.read()
            # img_ori = cv2.rotate(img_ori, cv2.ROTATE_90_CLOCKWISE)
            if letterbox_resize:
                img, resize_ratio, dw, dh = letterbox_resize_f(img_ori, *new_size)
            else:
                height_ori, width_ori = img_ori.shape[:2]
                img = cv2.resize(img_ori, new_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img, np.float32)
            img = img[np.newaxis, :] / 255.

            boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

            # rescale the coordinates to the original image
            if letterbox_resize:
                boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
                boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
            else:
                boxes_[:, [0, 2]] *= (width_ori / float(new_size[0]))
                boxes_[:, [1, 3]] *= (height_ori / float(new_size[1]))

            for bi in range(len(boxes_)):
                if classes[labels_[bi]] == 'person':
                    x0, y0, x1, y1 = boxes_[bi]
                    yield [i, -1, x0, y0, x1, y1, scores_[bi], -1, -1, -1]

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    end = time()
    print(
        f"Detection done in {end - start}s, {(end - start) / video_frame_count}s/f = {video_frame_count / (end - start)}f/s")
    video.release()

