from time import time

import os
import sacred

from src.exp.main import people_detection

name = "compute_detections"
ex = sacred.Experiment(name)
ex.observers.append(sacred.observers.FileStorageObserver(f'runs/{name}'))
ex.add_config('config.json')


@ex.config
def config_1():
    detect_people_params = dict(anchor_path='YOLOv3_TensorFlow/data/yolo_anchors.txt',
                                class_name_path='YOLOv3_TensorFlow/data/coco.names',
                                restore_path='YOLOv3_TensorFlow/data/darknet_weights/yolov3.ckpt',
                                # letterbox resize==keep aspect ratio
                                letterbox_resize=True,
                                # 16:9
                                new_size=[1280, 720],
                                )


@ex.automain
def main(video_directory, detect_people_params):
    assert os.path.exists(video_directory), video_directory

    for directory, _, filenames in os.walk(video_directory):
        print(f'Scanning {directory}')
        for file in filenames:
            path = os.path.join(directory, file)
            tmp_output = f'/tmp/{time()}_detections.txt'
            with open(tmp_output, "w") as f:
                for d in people_detection(video=path, **detect_people_params):
                    print(*d, sep=",", file=f)
            ex.add_artifact(tmp_output, "detections.txt")
