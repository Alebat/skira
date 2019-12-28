import unittest
from argparse import Namespace

from src.bboxes import crop, highlight
from src.main import main


class MyTestCase(unittest.TestCase):
    def test_something1(self):
        crop('../data/vids/IMG_0818.MOV', '../data/vids/Ski-18_iou.txt', '../data/output/crop_test_0818', 'mov',
             False, True)

    def test_something2(self):
        crop('../data/vids/IMG_0819.MOV', '../data/vids/Ski-19_iou.txt', '../data/output/crop_test_0819', 'mov',
             False, True)

    def test_something2_1(self):
        highlight('../data/vids/IMG_0819.MOV', '../data/vids/Ski-19_iou.txt', '../data/output/crop_test_0819.mov',
                  False, True)

    def test_something2_2(self):
        # highlight('../data/vids/IMG_0887.MOV', '../data/1577561725/tracks_IMG_0887.txt', '../data/1577561725/IMG_0887.MOV',
        #           False, False)
        crop('../data/vids/IMG_0887.MOV', '../data/1577561725/tracks_IMG_0887.txt', '../data/1577561725/IMG_0887.MOV',
             'mov', False, False)

    def test_main(self):
        main(Namespace(anchor_path='../YOLOv3_TensorFlow/data/yolo_anchors.txt',
                       class_name_path='../YOLOv3_TensorFlow/data/coco.names',
                       name='IMG_0887',
                       letterbox_resize=True,
                       new_size=[416, 416],
                       restore_path='../YOLOv3_TensorFlow/data/darknet_weights/yolov3.ckpt',
                       sigma_h=0.5, sigma_iou=0.5, sigma_l=0, t_min=2))


if __name__ == '__main__':
    unittest.main()
