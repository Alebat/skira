from time import time

import cv2
import numpy as np
import os
from tqdm import tqdm

from C3D_tensorflow.extract_c3d_sports1M import extract_p3d
from src.exp.base_experiment import get_skira_exp
from src.exp.bboxes import read_images

ex = get_skira_exp("extract_p3d")


@ex.config
def config_1():
    pass


@ex.automain
def main(videos_directory):
    test_videos = list(
        map(
            lambda x: os.path.join(videos_directory, x),
            filter(lambda x: x[-4:] == '.mov',
                   os.listdir(videos_directory)
                   )
        )
    )

    named_videos = [
        (read_images(cv2.VideoCapture(path)), path)
        for path in test_videos
    ]

    tmp_output = f'/tmp/{time()}_p3d_feats.npy'

    features = extract_p3d(40, tqdm(named_videos, total=len(test_videos)), augment=True)

    def save(current, file, tmp_output, videos_directory):
        if current is not None:
            print()
            print("Extracted ", current)
            arr = np.array(file)
            print("Saving ", arr.shape)
            np.save(tmp_output, arr)
            common = os.path.commonprefix([videos_directory, current])
            file_id = current[len(common):].replace("/", "_")
            ex.add_artifact(tmp_output, f'p3d-{file_id}.npy')

    file = None
    current = None
    for prediction, descriptor in features:
        if descriptor[0] != current:
            save(current, file, tmp_output, videos_directory)
            current = descriptor[0]
            file = []
        file.append(prediction.cpu().numpy().flatten())
    save(current, file, tmp_output, videos_directory)
