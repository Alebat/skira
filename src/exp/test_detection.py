from time import time

import os

from src.exp.base_experiment import get_skira_exp
from src.exp.main import people_detection

ex = get_skira_exp("test_detection")


@ex.named_config
def config_1():
    detect_people_params = dict(anchor_path='YOLOv3_TensorFlow/data/yolo_anchors.txt',
                                class_name_path='YOLOv3_TensorFlow/data/coco.names',
                                restore_path='YOLOv3_TensorFlow/data/darknet_weights/yolov3.ckpt',
                                letterbox_resize=True,
                                new_size=[1280, 720],
                                )

    video = 'Fondo Piccolo/IMG_0990.MOV'


@ex.automain
def test_detection(video_directory, video, detect_people_params):
    tmp_output = f'/tmp/{time()}_detections.txt'

    path = os.path.join(video_directory, video)
    assert os.path.exists(path), path

    with open(tmp_output, "w") as f:
        for d in people_detection(video=path, **detect_people_params):
            print(*d, sep=",", file=f)

    ex.add_artifact(tmp_output, "detections.txt")

