from time import time

import cv2
import os
import pandas as pd
import sacred
from tqdm import tqdm

from iou_tracker.util import save_to_csv
from src.exp.main import iou_mom_tracking

name = "compute_tracks"
ex = sacred.Experiment(name)
ex.observers.append(sacred.observers.FileStorageObserver(f'runs/{name}'))
ex.add_config('config.json')


@ex.config
def config_1():
    tracking_iou_mom_params = dict(
        # High threshold for confidence (0 = disabled, keep everything)
        sigma_l=0,
        # High threshold for confidence (0 = disabled, keep everything)
        sigma_h=0.5,
        # Low threshold for intersection over union
        sigma_iou=0.5,
        # Momentum smoothing (memory) factor
        mom_alpha=0.95,
        # Exponential zooming per frame factor
        exp_zoom=1.001,
        # Lower threshold for detection area
        min_area=20000,
        # Minimum track length (seconds)
        t_min=3,
        # Patience in seconds
        ttl=1/6,
    )
    resume = None


def record(output_file, stream):
    with open(output_file, "w") as f:
        for d in stream:
            print(*d, sep=",", file=f)
            yield d


@ex.automain
def main(video_directory, detections_directory, tracking_iou_mom_params, resume):
    assert os.path.exists(video_directory), video_directory
    print(f'Scanning root {video_directory}')
    files = []
    for directory, _, filenames in os.walk(video_directory):
        print(f'Scanning {directory}')
        for file in filenames:
            path = os.path.join(directory, file)
            files.append(path)

    total = len(files)

    if resume is not None:
        ind = files.index(resume)
        files = files[ind:]
        print(f'Resuming...')

    for file in tqdm(files, initial=total-len(files), total=total):
        common = os.path.commonprefix([video_directory, file])
        file_id = file[len(common):].replace("/", "_")

        detections_file = os.path.join(detections_directory, f'detections-{file_id}.txt')

        if not os.path.exists(detections_file):
            continue

        print()
        print("Working on:", file_id)
        print("Path:", file)

        tmp_output = f'/tmp/{time()}_tracks.txt'

        vc = cv2.VideoCapture(file)
        video_fps = vc.get(cv2.CAP_PROP_FPS)
        vc.release()

        # Tracking

        file = ex.open_resource(detections_file)
        detections = pd.read_csv(file,
                                 sep=',',
                                 header=None,
                                 names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y',
                                        'z'])

        tracks = iou_mom_tracking(detections, fps=video_fps, **tracking_iou_mom_params)

        save_to_csv(tmp_output, tracks)
        ex.add_artifact(tmp_output, f'tracks-{file_id}.txt')
