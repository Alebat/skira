from time import time

import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.utils.data as data
from scipy.stats import spearmanr as sr, pearsonr
from torch.autograd import Variable

from MS_LSTM.dataloader import videoDataset, transform
from MS_LSTM.model import get_scoring_model
from src.exp.base_experiment import get_skira_exp
from src.util.one_cycle import OneCycleLR

ex = get_skira_exp("train_ms_lstm")


@ex.config
def config_1():
    test_size = 100
    ground_truth = "data/selected/gt.txt"
    epochs = 60
    lr_range = (2e-3, 2e-2)
    model = 'scoring'
    text_filter=''


def train_shuffle(model, min_mse, max_corr, max_corrp, trainset, testset, models_dir, epochs, lr_range, time_n):
    round_max_spea = 0
    round_max_pear = 0
    round_min_mse = 200

    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    valLoader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

    # build the model
    scoring = get_scoring_model(model)
    if torch.cuda.is_available():
        scoring.cuda()

    optimizer = optim.Adam(params=scoring.parameters())
    scheduler = OneCycleLR(optimizer, epochs, lr_range)

    total_params = sum(p.numel() for p in scoring.parameters() if p.requires_grad)
    print("Total Params: " + str(total_params))

    for epoch in range(epochs):
        scheduler.step()
        print(f'Epoch: {epoch}, LR: {scheduler.get_lr()}')
        total_regr_loss = 0
        total_scoring_mse = 0
        total_sample = 0
        for i, (features, scores) in enumerate(trainLoader):  # get mini-batch
            if torch.cuda.is_available():
                features = Variable(features).cuda()
                scores = Variable(scores).cuda()
            logits, penal = scoring(features)
            scoring_mse = scoring.loss(logits, scores)
            if penal is None:
                regr_loss = scoring_mse
            else:
                regr_loss = scoring_mse + penal
            # new three lines are back propagation
            optimizer.zero_grad()
            regr_loss.backward()
            # nn.utils.clip_grad_norm(scoring.parameters(), 1.5)
            optimizer.step()
            total_regr_loss += regr_loss.data.item() * scores.shape[0]
            total_scoring_mse += scoring_mse.data.item() * scores.shape[0]
            total_sample += scores.shape[0]

        training_loss = total_regr_loss / total_sample
        training_mse = total_scoring_mse / total_sample
        print("Regression Loss: " + str(training_loss))
        print("Training MSE: " + str(training_mse))
        torch.save(scoring.state_dict(), os.path.join(models_dir, f'time-{time_n}-epoch-{epoch}.pt'))
        scoring.eval()
        val_pred = []
        val_sample = 0
        val_loss = 0
        val_truth = []
        for j, (features, scores) in enumerate(valLoader):
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
        val_pr, _ = pearsonr(val_truth.flatten(), val_pred.flatten())
        val_mse = val_loss / val_sample

        if val_mse < min_mse:
            torch.save(scoring.state_dict(), os.path.join(models_dir, f'mark_40attn.pt'))
        min_mse = min(min_mse, val_mse)
        max_corr = max(max_corr, val_sr)
        max_corrp = max(max_corrp, val_pr)
        round_min_mse = min(round_min_mse, val_mse)
        round_max_spea = max(val_sr, round_max_spea)
        round_max_pear = max(val_pr, round_max_pear)

        print(f'Val: MSE:     {val_mse}, Spearman Corr: {val_sr}, Pearson Corr: {val_pr}')
        print(f'Val: Min MSE: {min_mse}, Max Spearman:  {max_corr}, Max Pearson: {max_corrp}')
        scoring.train()

        if training_mse < val_mse / 5:
            print('Early stopping')
            break

    print('MSE: %.2f spearman: %.2f' % (round_min_mse, round_max_spea))
    return min_mse, max_corr, max_corrp

actual_filter = filter

@ex.automain
def main(directory, ground_truth, test_size, model, seed, epochs, lr_range, text_filter):
    random.seed(seed)
    assert os.path.exists(ground_truth), "ground_truth"
    tmp_tr = f"/tmp/train_dataset_{time()}.txt"
    tmp_te = f"/tmp/test_dataset_{time()}.txt"
    with open(ground_truth, "r") as annotations:
        train = open(tmp_tr, 'w')
        test = open(tmp_te, 'w')
        lines = list(filter(lambda x: text_filter in x, list(annotations)))
        random.shuffle(lines)
        for i, line in enumerate(lines):
            name, mark = line.strip().split(',')
            print('c3d-_output-' + '-'.join(name.split('-')[1:]), mark, sep=',', file=test if i < test_size else train)
        train.close()
        test.close()
    ex.add_artifact(tmp_tr, "train.txt")
    ex.add_artifact(tmp_te, "test.txt")

    suffixes = [
        ".npy",
        ".mirror.npy",
        ".lfps.npy",
        ".mirror.lfps.npy",
    ]

    trainset = videoDataset(root=directory,
                            label=tmp_tr, suffixes=suffixes, transform=transform, data=None, augmented=True)
    testset = videoDataset(root=directory,
                           label=tmp_te, suffixes='.npy', transform=transform, data=None)

    tmp_dir = f"/tmp/models_{time()}"
    os.mkdir(tmp_dir)

    min_mse = 200
    max_corr = 0
    max_corrp = 0
    for time_n in range(3):
        min_mse, max_corr, max_corrp = train_shuffle(model, min_mse, max_corr, max_corrp, trainset, testset, tmp_dir, epochs, lr_range, time_n)

    for f in os.listdir(tmp_dir):
        ex.add_artifact(os.path.join(tmp_dir, f))
