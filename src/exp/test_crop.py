from time import time

import os
import pandas as pd

from src.exp.base_experiment import get_skira_exp
from src.exp.bboxes import crop

ex = get_skira_exp("test_crop")


@ex.named_config
def config_1():
    tmp_dir = f'/tmp/{time()}_test_crop'

    video = "Salizzona/IMG_0886.MOV"

    os.mkdir(tmp_dir)

    tracks_file = 'runs/test_tracker_iou_mom/1/tracks.txt'

    crop_params = dict(
        output_video=os.path.join(tmp_dir, f'output'),
        output_videos_extension='mov',
        relative_bboxes=False,
        rotate90=False,
    )


@ex.named_config
def config_2():
    tmp_dir = f'/tmp/{time()}_test_crop'

    video = "Salizzona/IMG_0886.MOV"

    os.mkdir(tmp_dir)

    tracks_file = 'runs/test_tracking_iou_mom/1/tracks.txt'

    crop_params = dict(
        output_video=os.path.join(tmp_dir, f'output'),
        output_videos_extension='mov',
        relative_bboxes=False,
        rotate90=False,
    )


@ex.named_config
def config_3():
    tmp_dir = f'/tmp/{time()}_test_crop'

    video = "Salizzona/IMG_0886.MOV"

    os.mkdir(tmp_dir)

    tracks_file = 'runs/test_tracking_iou_mom/2/tracks.txt'

    crop_params = dict(
        output_video=os.path.join(tmp_dir, f'output'),
        output_videos_extension='mov',
        relative_bboxes=False,
        rotate90=False,
    )


@ex.named_config
def config_4():
    tmp_dir = f'/tmp/{time()}_test_crop'

    video = "Fondo Piccolo/IMG_0990.MOV"

    os.mkdir(tmp_dir)

    tracks_file = 'runs/test_tracking_iou_mom/3/tracks.txt'

    crop_params = dict(
        output_video=os.path.join(tmp_dir, f'output'),
        output_videos_extension='mov',
        relative_bboxes=False,
        rotate90=False,
    )


@ex.automain
def test_crop(video_directory, video, tracks_file, crop_params, tmp_dir):
    path = os.path.join(video_directory, video)
    assert os.path.exists(path), path

    tracks_file = ex.open_resource(tracks_file)

    input_tracks = pd.read_csv(tracks_file,
                               sep=',',
                               header=None,
                               index_col=[0, 1],
                               names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x',
                                      'y', 'z'])

    crop(
        input_tracks=input_tracks,
        input_video=path,
        **crop_params
    )

    for file in os.listdir(tmp_dir):
        ex.add_artifact(os.path.join(tmp_dir, file), file)
