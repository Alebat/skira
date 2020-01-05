import csv

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


def read_images(vc, rotate90=False):
    yes = True
    f = 0
    while yes:
        yes, img = vc.read()
        if yes:
            if rotate90:
                yield f, cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            else:
                yield f, img
            f += 1


class SkiSequence(Dataset):
    def __init__(self, video, detections):
        self._det = detections if not isinstance(detections, str) \
            else pd.read_csv(detections,
                             sep=',',
                             header=None,
                             index_col=0,
                             names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])
        self.no_gt = True
        self._video = video
        self._transform = ToTensor()
        self._data, self._len = self._sequence()

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        """Return the ith image converted to blob"""

        f, img = next(self._data)

        if f in self._det.index:
            dets = self._det.loc[[f]]
            conv = np.array([det.astype(np.float32) for det in dets.values[:, 1:5]])
            conf = np.array([det.astype(np.float32) for det in dets.values[:, 5]])
        else:
            conv = np.ndarray((0,))
            conf = np.ndarray((0,))

        sample = {
            'img': self._transform(img),
            'dets': conv,
            'conf': conf,
            'index': f,
        }

        return sample

    def _sequence(self):
        vc = cv2.VideoCapture(self._video)
        length = int(vc.get(7))
        images = read_images(vc)
        return images, length

    def __str__(self):
        return f"SkiSequence-{self._video}"

    @staticmethod
    def write_results(all_tracks, output_file):
        """Write the tracks in the format for MOT16/MOT17 sumbission

        all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key
         track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """

        with open(output_file, "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    writer.writerow([frame + 1, i + 1, x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, -1, -1, -1, -1])
