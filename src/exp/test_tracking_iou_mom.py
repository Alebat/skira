from time import time

import os
import pandas as pd

from iou_tracker.util import save_to_csv
from src.exp.base_experiment import get_skira_exp
from src.exp.bboxes import highlight
from src.exp.main import iou_mom_tracking

ex = get_skira_exp("test_tracking_iou_mom")


@ex.named_config
def config_1():
    detections_file = 'data/1578309853_custom_res_det/det_IMG_0886_custom_res_det.txt'
    tracking_iou_mom_params = dict(
        sigma_l=0,
        sigma_h=0.5,
        sigma_iou=0.6,
        t_min=59,
        ttl=10,
        mom_alpha=0.95,
        exp_zoom=1.001,
    )
    video = 'Salizzona/IMG_0886.MOV'


@ex.named_config
def config_2():
    detections_file = 'data/1578309853_custom_res_det/det_IMG_0886_custom_res_det.txt'
    tracking_iou_mom_params = dict(
        sigma_l=0,
        sigma_h=0.5,
        sigma_iou=0.6,
        t_min=59,
        ttl=10,
        mom_alpha=0.95,
        exp_zoom=1.001,
        min_area=10000
    )
    video = 'Salizzona/IMG_0886.MOV'


@ex.named_config
def config_3():
    detections_file = 'runs/test_detection/5/detections.txt'
    tracking_iou_mom_params = dict(
        sigma_l=0,
        sigma_h=0.5,
        sigma_iou=0.6,
        t_min=59,
        ttl=10,
        mom_alpha=0.95,
        exp_zoom=1.001,
        min_area=10000
    )
    video = 'Fondo Piccolo/IMG_0990.MOV'


@ex.automain
def test_tracker_iou_mom(video_directory, video, detections_file, tracking_iou_mom_params):
    tmp_output = f'/tmp/{time()}_tracking_iou_mom_highlighted.mov'
    tmp_output_t = f'/tmp/{time()}_tracking_iou_mom_tracks.txt'

    path = os.path.join(video_directory, video)
    assert os.path.exists(path), path

    file = ex.open_resource(detections_file)
    detections = pd.read_csv(file,
                             sep=',',
                             header=None,
                             names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])

    tracks = iou_mom_tracking(detections, **tracking_iou_mom_params)

    save_to_csv(tmp_output_t, tracks)
    ex.add_artifact(tmp_output_t, "tracks.txt")

    highlight(path, tmp_output_t,
              tmp_output, False, False)

    ex.add_artifact(tmp_output, "output.mov")
