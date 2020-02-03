from time import time

import numpy as np
import os
import random
import torch
import torch.utils.data as data
from torch.autograd import Variable

from MS_LSTM.dataloader import videoDataset, transform
from MS_LSTM.model import Scoring
from src.exp.base_experiment import get_skira_exp

ex = get_skira_exp("test_ms_lstm")


@ex.config
def config_1():
    test_size = 50
    ground_truth = "data/selected/gt.txt"




def test_shuffle(testset, model):
    testLoader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

    # build the model
    scoring = Scoring(feature_size=4096)
    if torch.cuda.is_available():
        scoring.cuda()

    scoring.load_state_dict(torch.load(model))
    scoring.eval()
    for epoch in range(1):  # total 40 epoches
        scoring.eval()
        val_pred = []
        val_sample = 0
        val_loss = 0
        val_truth = []
        for j, (features, scores) in enumerate(testLoader):
            val_truth.append(scores.numpy())
            if torch.cuda.is_available():
                features = Variable(features).cuda()
                scores = Variable(scores).cuda()
            regression, _ = scoring(features)
            val_pred.append(regression.data.cpu().numpy())
            regr_loss = scoring.loss(regression, scores)
            val_loss += (regr_loss.data.item()) * scores.shape[0]
            val_sample += scores.shape[0]
        val_truth = np.concatenate(val_truth)
        val_pred = np.concatenate(val_pred)
    for i in range(val_truth.shape[0]):
        print('GT: ' + str(val_truth[i]) + '\t' + "Pred: " + str(val_pred[i]) + 'Res: ' + str(val_truth[i]-val_pred[i]))


@ex.automain
def main(directory, ground_truth, test_size, seed, model):
    random.seed(seed)
    assert os.path.exists(ground_truth), "ground_truth"
    tmp_te = f"/tmp/test_dataset_{time()}.txt"
    with open(ground_truth, "r") as annotations:
        test = open(tmp_te, 'w')
        lines = list(annotations)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            name, mark = line.strip().split(',')
            if i < test_size:
                print('c3d-_output-' + '-'.join(name.split('-')[1:]), mark, sep=',', file=test)
        test.close()
    ex.add_artifact(tmp_te)

    testset = videoDataset(root=directory,
                           label=tmp_te, suffixes='.npy', transform=transform, data=None)

    tmp_dir = f"/tmp/models_{time()}"
    os.mkdir(tmp_dir)

    test_shuffle(testset, model)
