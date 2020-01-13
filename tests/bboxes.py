import unittest
from argparse import Namespace
from time import time

import os
import pandas as pd
import sacred

from src.exp.bboxes import crop, highlight
from src.exp.main import main, wobnw_tracking, iou_mom_tracking
from src.exp.main import people_detection

ex = sacred.Experiment("uno")
ex.observers.append(sacred.observers.FileStorageObserver("runs"))


class MyTestCase(unittest.TestCase):
    def test_something1(self):
        crop('../data/vids/IMG_0818.MOV', '../data/vids/Ski-18_iou.txt', '../data/output/crop_test_0818', 'mov',
             False, True)

    def test_something2(self):
        crop('../data/vids/IMG_0819.MOV', '../data/vids/Ski-19_iou.txt', '../data/output/crop_test_0819', 'mov',
             False, True)

    def test_something2_1(self):
        highlight('../data/vids/IMG_0886.MOV', '../data/1578220812/tracks_IMG_0886_hi_res_det.txt',
                  '../data/1578220812/main_IMG_0886.MOV', False, False)

    def test_something2_2(self):
        # highlight('../data/vids/IMG_0886.MOV', '../data/1577561725/tracks_IMG_0886.txt',
        #           '../data/1577561725/IMG_0886.MOV',
        #           False, False)
        crop('../data/vids/IMG_0886.MOV', '../data/1577561725/tracks_IMG_0886.txt', '../data/1577561725/IMG_0886.MOV',
             'mov', False, False)

    def test_main(self):
        main(Namespace(anchor_path='../YOLOv3_TensorFlow/data/yolo_anchors.txt',
                       class_name_path='../YOLOv3_TensorFlow/data/coco.names',
                       name='IMG_0886_std_res_det',
                       video='../data/vids/IMG_0886.MOV',
                       letterbox_resize=True,
                       new_size=[416, 416],
                       restore_path='../YOLOv3_TensorFlow/data/darknet_weights/yolov3.ckpt',
                       sigma_l=0, sigma_h=0.5, sigma_iou=0.6, t_min=59))

    def test_main_hi_res(self):
        main(Namespace(anchor_path='../YOLOv3_TensorFlow/data/yolo_anchors.txt',
                       class_name_path='../YOLOv3_TensorFlow/data/coco.names',
                       name='IMG_0886_hi_res_det',
                       video='../data/vids/IMG_0886.MOV',
                       letterbox_resize=True,
                       new_size=[832, 416],
                       restore_path='../YOLOv3_TensorFlow/data/darknet_weights/yolov3.ckpt',
                       sigma_l=0, sigma_h=0.5, sigma_iou=0.6, t_min=59))

    def test_main_custom_res(self):
        main(Namespace(anchor_path='../YOLOv3_TensorFlow/data/yolo_anchors.txt',
                       class_name_path='../YOLOv3_TensorFlow/data/coco.names',
                       name='IMG_0886_custom_res_det',
                       video='../data/vids/IMG_0886.MOV',
                       letterbox_resize=True,
                       new_size=[1280, 720],
                       restore_path='../YOLOv3_TensorFlow/data/darknet_weights/yolov3.ckpt',
                       sigma_l=0, sigma_h=0.5, sigma_iou=0.6, t_min=59))

    def test_detection(self):
        time_id = int(time())
        os.makedirs(f'../data/{time_id}', exist_ok=False)
        name = 'IMG_0886'

        detections = people_detection(anchor_path='../YOLOv3_TensorFlow/data/yolo_anchors.txt',
                               class_name_path='../YOLOv3_TensorFlow/data/coco.names',
                               restore_path='../YOLOv3_TensorFlow/data/darknet_weights/yolov3.ckpt',
                               video='../data/vids/IMG_0886.MOV',
                               letterbox_resize=True,
                               new_size=[416, 416],
                               )
        with open(f'../data/{time_id}/det_{name}.txt', "w") as f:
            for d in detections:
                print(*d, sep=",", file=f)

    def test_detection_all(self):
        time_id = str(int(time())) + '_det_all_hres'
        os.makedirs(f'../data/{time_id}', exist_ok=False)

        path = '/media/ale/Volume/SkiVideos'
        print(f'Scanning path: {path}')
        dirs = os.listdir(path)
        dirs.sort()
        for d in dirs:
            print(f'Scanning dir: {d}')
            files = os.listdir(os.path.join(path, d))
            files.sort()
            for f in files:
                name = os.path.basename(f).split('.')[0]
                detections = people_detection(anchor_path='../YOLOv3_TensorFlow/data/yolo_anchors.txt',
                                              class_name_path='../YOLOv3_TensorFlow/data/coco.names',
                                              video=os.path.join(path, d, f),
                                              letterbox_resize=True,
                                              new_size=[1024, 576],
                                              restore_path='../YOLOv3_TensorFlow/data/darknet_weights/yolov3.ckpt',
                                              )

                with open(f'../data/{time_id}/det_{name}.txt', "w") as f:
                    for det in detections:
                        print(*det, sep=",", file=f)

    def test_tracking_wo_bnw(self):
        time_id = f'{int(time())}_wobnw_1578220812'
        os.makedirs(f'../data/{time_id}', exist_ok=False)
        name = 'IMG_0886'

        wobnw_tracking(video='../data/1578220812/main_IMG_0886.MOV',
                       detections='../data/1578220812/det_IMG_0886_hi_res_det.txt',
                       obj_detect_model='../tracking_wo_bnw/output/faster_rcnn_fpn_training_mot_17/model_epoch_27.model',
                       reid_weights="../tracking_wo_bnw/output/tracktor/reid/res50-mot17-batch_hard/ResNet_iter_25245"
                                    ".pth",
                       output_file=f'../data/{time_id}/tracks_{name}.txt')

    def test_tracker_iou_mom(self):
        time_id = f'iou_mom_mom_{int(time())}'

        detections = pd.read_csv('../data/1578309853_custom_res_det/det_IMG_0886_custom_res_det.txt',
                                 sep=',',
                                 header=None,
                                 names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y',
                                        'z'])

        iou_mom_tracking(
            detections=detections,
            sigma_l=0,
            sigma_h=0.5,
            sigma_iou=0.6,
            t_min=59,
            ttl=10,
            mom_alpha=0.95,
            exp_zoom=1.001,
        )

        output = f'/tmp/{time()}_tracking_iou_mom_tracks.txt'
        ex.add_artifact(output, "tracks")

        # highlight('../data/vids/IMG_0886.MOV', f'../data/{time_id}/tracks_IMG_0886_iou_mom_mom.txt',
        #           f'../data/{time_id}/main_IMG_0886.MOV', False, False)


if __name__ == '__main__':
    unittest.main()
