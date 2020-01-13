from time import time

import cv2
import os
import sacred
from tqdm import tqdm

from iou_tracker.util import save_to_csv
from src.exp.bboxes import read_images
from src.exp.main import iou_mom_tracking, PeopleDetection

name = "compute_detections_and_tracks"
ex = sacred.Experiment(name)
ex.observers.append(sacred.observers.FileStorageObserver(f'runs/{name}'))
ex.add_config('config.json')


@ex.config
def config_1():
    detect_people_params = dict(
        anchor_path='YOLOv3_TensorFlow/data/yolo_anchors.txt',
        class_name_path='YOLOv3_TensorFlow/data/coco.names',
        restore_path='YOLOv3_TensorFlow/data/darknet_weights/yolov3.ckpt',
        # letterbox resize==keep aspect ratio
        letterbox_resize=True,
        # 16:9
        new_size=[1280, 720],
    )
    tracking_iou_mom_params = dict(
        # High threshold for confidence (0 = disabled, keep everything)
        sigma_l=0,
        # High threshold for confidence (0 = disabled, keep everything)
        sigma_h=0.5,
        # Low threshold for intersection over union
        sigma_iou=0.6,
        # Momentum smoothing (memory) factor
        mom_alpha=0.95,
        # Exponential zooming per frame factor
        exp_zoom=1.001,
        # Lower threshold for detection area
        min_area=10000,
        # Minimum track length (seconds)
        t_min=1,
        # Patience in seconds
        ttl=1/6,
    )


def record(output_file, stream):
    with open(output_file, "w") as f:
        for d in stream:
            print(*d, sep=",", file=f)
            yield d


@ex.automain
def main(video_directory, detect_people_params, tracking_iou_mom_params):
    assert os.path.exists(video_directory), video_directory
    print(f'Scanning root {video_directory}')
    files = []
    for directory, _, filenames in os.walk(video_directory):
        print(f'Scanning {directory}')
        for file in filenames:
            path = os.path.join(directory, file)
            files.append(path)

    for file in tqdm(files):
        with PeopleDetection(**detect_people_params) as detector:
            print()

            common = os.path.commonprefix([video_directory, file])
            file_id = file[len(common):].replace("/", "_")

            print("Working on:", file_id)

            tmp_output = f'/tmp/{time()}_detections.txt'
            tmp_output_t = f'/tmp/{time()}_tracks.txt'

            # Detection

            vc = cv2.VideoCapture(file)
            video_fps = vc.get(cv2.CAP_PROP_FPS)
            video_frames_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

            detections = record(
                tmp_output,
                detector.process([img for f, img in read_images(vc)], video_frames_count)
            )

            # Tracking

            tracks = iou_mom_tracking(list(detections), fps=video_fps, **tracking_iou_mom_params)

            save_to_csv(tmp_output_t, tracks)

            ex.add_artifact(tmp_output, f'detections-{file_id}.txt')
            ex.add_artifact(tmp_output_t, f'tracks-{file_id}.txt')
