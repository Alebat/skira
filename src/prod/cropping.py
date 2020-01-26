from time import time

import os
import pandas as pd

from src.exp.base_experiment import get_skira_exp
from src.exp.bboxes import crop

ex = get_skira_exp("cropping")


@ex.config
def config_1():
    tmp_dir = f'/tmp/{time()}_cropping'
    os.mkdir(tmp_dir)

    crop_params = dict(
        output_videos_extension='mov',
        relative_bboxes=False,
        rotate90=False,
    )

    tracks_directory = ''
    resume = None


@ex.automain
def main(video_directory, crop_params, tracks_directory, tmp_dir, resume):
    assert os.path.exists(tracks_directory), tracks_directory
    print(f'Scanning root {tracks_directory}')
    t_files = os.listdir(tracks_directory)
    video_ids = [os.path.basename(x) for x in t_files]

    assert os.path.exists(video_directory), video_directory
    print(f'Scanning root {video_directory}')
    files = []
    for directory, _, filenames in os.walk(video_directory):
        print(f'Scanning {directory}')
        for file in filenames:
            path = os.path.join(directory, file)
            files.append(path)

    if resume is not None:
        ind = files.index(resume)
        files = files[ind:]
        print(f'Resuming...')

    for n, file in enumerate(files):
        common = os.path.commonprefix([video_directory, file])
        file_id = file[len(common):].replace("/", "_")
        tracks_filename = f'tracks-{file_id}.txt'
        if tracks_filename in video_ids:
            print(f'Video {n} of {len(files)}')

            tracks_path = os.path.join(tracks_directory, tracks_filename)

            if os.path.exists(tracks_path):
                print("Working on:", file, "with", tracks_path)

                tracks_file = ex.open_resource(tracks_path)
                input_tracks = pd.read_csv(tracks_file,
                                           sep=',',
                                           header=None,
                                           index_col=[0, 1],
                                           names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf',
                                                  'x', 'y', 'z'])

                crop(
                    output_video=os.path.join(tmp_dir, f'output-{file_id}'),
                    input_tracks=input_tracks,
                    input_video=file,
                    **crop_params
                )

                for output_file in os.listdir(tmp_dir):
                    ex.add_artifact(os.path.join(tmp_dir, output_file), output_file)
            else:
                print("Does not exist:", tracks_path)
