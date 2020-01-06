import os
from time import time

import pandas as pd
import sacred

from iou_tracker.util import save_to_csv
from src.bboxes import highlight
from src.main import iou_tracking, iou_mom_tracking
from src.main import people_detection

ex = sacred.Experiment("test_tracker_iou_mom")
ex.observers.append(sacred.observers.FileStorageObserver("runs/test_tracker_iou_mom"))
ex.add_config('config.json')


@ex.capture
def detection(**kwargs):
    ex.add_resource(kwargs['video'])
    people_detection(**kwargs)


@ex.capture
def tracking_iou(**kwargs):
    ex.add_resource(kwargs['video'])
    iou_tracking(**kwargs)


@ex.capture
def tracking_iou_mom(detections_file, sigma_l, sigma_h, sigma_iou, t_min, ttl, mom_alpha, exp_zoom):
    file = ex.open_resource(detections_file)
    detections = pd.read_csv(file,
                             sep=',',
                             header=None,
                             names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])
    tracks = iou_mom_tracking(detections, sigma_l, sigma_h, sigma_iou, t_min, ttl, mom_alpha, exp_zoom)

    output = f'/tmp/{time()}_tracking_iou_mom_tracks.txt'
    save_to_csv(output, tracks)
    ex.add_artifact(output, "tracks.txt")
    return output, tracks


@ex.capture
def highlighting(**kwargs):
    people_detection(**kwargs)


@ex.capture
def cropping(**kwargs):
    people_detection(**kwargs)


@ex.automain
def test_tracker_iou_mom(videos):

    tracks_file, tracks = tracking_iou_mom(
        detections_file='data/1578309853_custom_res_det/det_IMG_0886_custom_res_det.txt',
        sigma_l=0,
        sigma_h=0.5,
        sigma_iou=0.6,
        t_min=59,
        ttl=10,
        mom_alpha=0.95,
        exp_zoom=1.001,
    )

    tmp_output = f'/tmp/{time()}_tracking_iou_mom_highlighted.mov'

    video = os.path.join(videos, 'Salizzona/IMG_0886.MOV')
    assert os.path.exists(video), video

    highlight(video, tracks_file,
              tmp_output, False, False)

    ex.add_artifact(tmp_output, "highlighted.mov")
