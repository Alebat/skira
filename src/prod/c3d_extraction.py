from time import time

import cv2
import numpy as np
import os
import sacred
from tqdm import tqdm

from C3D_tensorflow.extract_c3d_sports1M import BATCH_SIZE, extract_c3d
from src.exp.bboxes import read_images

name = "extract_c3d_fc6"
ex = sacred.Experiment(name)
ex.observers.append(sacred.observers.FileStorageObserver(f'runs/{name}'))
ex.add_config('config.json')


@ex.config
def config_1():
    model_name = "data/conv3d_deepnetA_sport1m_iter_1900000_TF.model"
    videos_directory = "6"
    device = '/gpu:0'


@ex.automain
def main(model_name, videos_directory, device):
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

    tmp_output = f'/tmp/{time()}_c3d_feats.npy'

    features = extract_c3d(BATCH_SIZE, device, model_name, tqdm(named_videos, total=len(test_videos)))

    file = []
    curr = None
    for prediction, descriptor in features:
        if descriptor[0] != curr:
            if curr is not None:
                np.save(tmp_output, np.array(file))
                common = os.path.commonprefix([videos_directory, curr])
                file_id = curr[len(common):].replace("/", "_")
                ex.add_artifact(tmp_output, f'detections-{file_id}.npy')

            file = []
            curr = descriptor[0]
            print("Extracting from", curr)
        file.append(prediction)
