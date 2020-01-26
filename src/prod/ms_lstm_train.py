from time import time

import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.utils.data as data
from scipy.stats import spearmanr as sr
from torch.autograd import Variable
from torch.optim import lr_scheduler

from MS_LSTM.dataloader import videoDataset, transform
from MS_LSTM.model import Scoring
from src.exp.base_experiment import get_skira_exp

ex = get_skira_exp("train_ms_lstm")


@ex.config
def config_1():
    test_size = 100
    ground_truth = "data/selected/gt.txt"


def train_shuffle(min_mse, max_corr, trainset, testset, models_dir):
    round_max_spea = 0
    round_min_mse = 200

    trainLoader = torch.utils.data.DataLoader(trainset,
                                              batch_size=128, shuffle=True, num_workers=0)
    testLoader = torch.utils.data.DataLoader(testset,
                                             batch_size=64, shuffle=False, num_workers=0)

    # build the model
    scoring = Scoring(feature_size=4096)
    if torch.cuda.is_available():
        scoring.cuda()

    total_params = sum(p.numel() for p in scoring.parameters() if p.requires_grad)
    print("Total Params: " + str(total_params))
    optimizer = optim.Adam(params=scoring.parameters(), lr=0.0005)  # use SGD optimizer to optimize the loss function
    scheduler = lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.7)
    for epoch in range(500):  # total 40 epoches
        # scheduler.step()
        print("Epoch:  " + str(epoch) + "Total Params: %d" % total_params)
        total_regr_loss = 0
        total_sample = 0
        for i, (features, scores) in enumerate(trainLoader):  # get mini-batch
            # print("%d batches have done" % i)
            if torch.cuda.is_available():
                features = Variable(features).cuda()
                scores = Variable(scores).cuda()
            # regression, logits = scoring(features)
            logits, penal = scoring(features)
            if penal is None:
                regr_loss = scoring.loss(logits, scores)
            else:
                regr_loss = scoring.loss(logits, scores) + penal
            # new three lines are back propagation
            optimizer.zero_grad()
            regr_loss.backward()
            # nn.utils.clip_grad_norm(scoring.parameters(), 1.5)
            optimizer.step()
            total_regr_loss += regr_loss.data.item() * scores.shape[0]
            total_sample += scores.shape[0]

        print("Classification Loss: " + str(total_regr_loss / total_sample))
        # the rest is used to evaluate the model with the test dataset
        torch.save(scoring.state_dict(), os.path.join(models_dir, f'epoch{epoch}.pt'))
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
        val_sr, _ = sr(val_truth, val_pred)
        if val_loss / val_sample < min_mse:
            torch.save(scoring.state_dict(), os.path.join(models_dir, f'mark_40attn.pt'))
        min_mse = min(min_mse, val_loss / val_sample)
        max_corr = max(max_corr, val_sr)
        round_min_mse = min(round_min_mse, val_loss / val_sample)
        round_max_spea = max(val_sr, round_max_spea)
        print("Val Loss: %.2f Correlation: %.2f Min Val Loss: %.2f Max Correlation: %.2f" %
              (val_loss / val_sample, val_sr, min_mse, max_corr))
        scoring.train()
    print('MSE: %.2f spearman: %.2f' % (round_min_mse, round_max_spea))
    return min_mse, max_corr


@ex.automain
def main(directory, ground_truth, test_size, seed):
    random.seed(seed)
    assert os.path.exists(ground_truth), "ground_truth"
    tmp_tr = f"/tmp/train_dataset_{time()}.txt"
    tmp_te = f"/tmp/test_dataset_{time()}.txt"
    with open(ground_truth, "r") as annotations:
        train = open(tmp_tr, 'w')
        test = open(tmp_te, 'w')
        lines = list(annotations)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            name, mark = line.strip().split(',')
            print('c3d-_output-' + '-'.join(name.split('-')[1:]), mark, sep=',', file=test if i < test_size else train)
        train.close()
        test.close()
    ex.add_artifact(tmp_tr)
    ex.add_artifact(tmp_te)

    trainset = videoDataset(root=directory,
                            label=tmp_tr, suffix=".npy", transform=transform, data=None)
    testset = videoDataset(root=directory,
                           label=tmp_te, suffix='.npy', transform=transform, data=None)

    tmp_dir = f"/tmp/models_{time()}"
    os.mkdir(tmp_dir)

    min_mse = 200
    max_corr = 0
    for _ in range(5):
        min_mse, max_corr = train_shuffle(min_mse, max_corr, trainset, testset, tmp_dir)

    for f in os.listdir(tmp_dir):
        ex.add_artifact(os.path.join(tmp_dir, f))
