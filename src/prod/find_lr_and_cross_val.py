from time import time

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import shutil
import torch
import torch.optim as optim
import torch.utils.data as data
from scipy.stats import spearmanr as sr
from torch.autograd import Variable

from MS_LSTM.dataloader import videoDataset, transform
from MS_LSTM.model import get_scoring_model
from src.exp.base_experiment import get_skira_exp
from src.util.lr_finder import LRFinder
from src.util.one_cycle import OneCycleLR

ex = get_skira_exp("4_find_lr_and_cross_val")


def rm_r(path):
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


@ex.config
def config_1():
    max_epochs = 60
    model = 'scoring'
    no_nfps = False
    no_lfps = False
    no_flip = False
    featn = 4096
    k_cross_val = 5
    ground_truth = None
    train_set = None
    test_set = None
    seed = 29736184


def run(model, trainset, testset, models_dir, epochs, lr_peak, time_n, features):
    max_spea = 0
    min_mae = 1000
    min_mse = 1000

    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    valLoader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)
    # build the model
    scoring = get_scoring_model(model, features)
    if torch.cuda.is_available():
        scoring.cuda()

    optimizer = optim.Adam(params=scoring.parameters())
    scheduler = OneCycleLR(optimizer, epochs, (lr_peak / 100, lr_peak / 10))

    total_params = sum(p.numel() for p in scoring.parameters() if p.requires_grad)
    print("Total Params: " + str(total_params))

    for epoch in range(epochs):
        scheduler.step()
        print(f'Epoch: {epoch}')
        print(f'LR: {scheduler.get_lr()}')
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

        # validation
        scoring.eval()
        val_sample = 0
        val_loss = 0
        val_truth = []
        val_pred = []
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
        val_mae = np.mean(np.abs(val_truth - val_pred))
        val_mse = val_loss / val_sample

        if True or val_mse < min_mse <= 1.5 or 0.7 <= max_spea < val_sr:
            torch.save(scoring.state_dict(), os.path.join(models_dir, f'time-{time_n}-epoch-{epoch}.pt'))

            with open(os.path.join(models_dir, f'time-{time_n}-epoch-{epoch}.csv'), 'w') as f:
                for i in range(val_truth.shape[0]):
                    print(val_truth[i][0], val_pred[i][0], sep=',', file=f)

            plt.scatter(val_truth, val_pred)
            plt.xlabel("Truth")
            plt.ylabel("Predicted")
            plt.savefig(os.path.join(models_dir, f'time-{time_n}-epoch-{epoch}_plot.pdf'))
            plt.clf()

            gb = pd.DataFrame(np.array([val_truth.flatten(), val_pred.flatten()]).T).groupby(0)
            gb = np.array([[i[0], list(i[1][1])] for i in gb])
            labels = gb[:, 0]
            gb = gb[:, 1]

            plt.boxplot(gb, labels=labels)
            plt.xlabel("Truth")
            plt.ylabel("Predicted")
            plt.savefig(os.path.join(models_dir, f'time-{time_n}-epoch-{epoch}_boxplot.pdf'))

        min_mae = min(min_mae, val_mae)
        min_mse = min(min_mse, val_mse)
        max_spea = max(max_spea, val_sr)

        print(f'ValMSE: {val_mse}', f'SpearmanCorr: {val_sr}', sep='\n')
        scoring.train()

        if training_mse < val_mse / 3:
            print('Early stopping')
            break

    return min_mae, min_mse, max_spea


@ex.automain
def main(directory, ground_truth, train_set, test_set, model, seed, max_epochs, no_nfps, k_cross_val, featn, no_lfps, no_flip):
    tmp_dir = "/tmp/skira"
    rm_r(tmp_dir)
    os.mkdir(tmp_dir)
    random.seed(seed)

    is_cv = k_cross_val > 1

    if is_cv:
        assert train_set is None and test_set is None, "Either ground_truth for CV or train_set and test_set for test"
        assert os.path.exists(ground_truth), "ground_truth"
    else:
        assert ground_truth is None, "Either ground_truth for CV or train_set and test_set for test"
        assert os.path.exists(train_set), "ground_truth"
        assert os.path.exists(test_set), "ground_truth"

    with open(train_set, "r") as annotations:
        lines = list(annotations)
        random.shuffle(lines)

    prefix = 'c3d-_output-' if 'c3d' in directory else 'p3d-_output-'
    suffixes = []

    if not no_nfps:
        suffixes.append('.npy')
    if not no_flip and not no_nfps:
        suffixes.append('.flip.npy')
    if not no_lfps:
        suffixes.append('.lfps.npy')
    if not no_lfps and not no_flip:
        suffixes.append('.flip.lfps.npy')

    tmp_tr = f"{tmp_dir}/lr_find_dataset_{time()}.txt"
    tmp_pl = f"{tmp_dir}/lr_find_plot_{time()}.pdf"

    with open(tmp_tr, 'w') as train:
        for i, line in enumerate(lines):
            name, mark = line.strip().split(',')
            print(prefix + '-'.join(name.split('-')[1:]), mark, sep=',', file=train)

    ex.add_artifact(tmp_tr, f"lr_find_dataset.txt")

    trainset = videoDataset(root=directory,
                            label=tmp_tr, suffixes=suffixes, transform=transform, data=None)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    # build the model
    scoring = get_scoring_model(model, featn=featn)
    if torch.cuda.is_available():
        scoring.cuda()

    optimizer = optim.Adam(params=scoring.parameters())  # use SGD optimizer to optimize the loss function

    class PLoss(torch.nn.Module):
        def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            logits, penal = input
            if penal is None:
                return scoring.loss(logits, target)
            else:
                return scoring.loss(logits, target) + penal

    lr_finder = LRFinder(scoring, optimizer, PLoss(), device="cuda")
    lr_finder.range_test(train_loader, end_lr=100, num_iter=100)

    lrs = lr_finder.history["lr"]
    losses = lr_finder.history["loss"]

    lr_argmin = lrs[np.argmin(losses)]

    print('Min found in', lr_argmin)

    lr_finder.plot().savefig(tmp_pl)

    ex.add_artifact(tmp_pl, "lr_find_plot.pdf")

    # actual training

    lines = []
    with open(train_set, "r") as f:
        lines.extend(f)
    random.shuffle(lines)

    tmp_dir = f"{tmp_dir}/models_{time()}"
    os.mkdir(tmp_dir)

    s_mse = 0
    s_corr = 0
    for time_n in range(k_cross_val):
        if is_cv:  # this is cross-val
            tmp_tr = f"{tmp_dir}/train_dataset_{time()}.txt"
            tmp_te = f"{tmp_dir}/validation_dataset_{time()}.txt"
            train = open(tmp_tr, 'w')
            validation = open(tmp_te, 'w')
            for i, line in enumerate(lines):
                name, mark = line.strip().split(',')
                print(prefix + '-'.join(name.split('-')[1:]), mark,
                      sep=',',
                      file=validation if i % k_cross_val == time_n else train)
            train.close()
            validation.close()

            ex.add_artifact(tmp_tr, f"{time_n}-train.txt")
            ex.add_artifact(tmp_te, f"{time_n}-validation.txt")
        else:  # this is test
            tmp_tr = train_set
            tmp_te = test_set

        trainset = videoDataset(root=directory, label=tmp_tr, suffixes=suffixes, transform=transform, data=None)
        valset = videoDataset(root=directory, label=tmp_te, suffixes=suffixes[0], transform=transform, data=None)

        min_mse, min_mae, max_corr = run(model, trainset, valset, tmp_dir, max_epochs, lr_argmin, time_n, featn)

        print('MinValMAE', min_mae)
        print('MinValMSE', min_mse)
        print('MaxSpearmanCorr', max_corr)
        s_mse += min_mse
        s_corr += max_corr

    print('AvgValMSE', s_mse / k_cross_val)
    print('AvgSpearmanCorr', s_corr / k_cross_val)

    for f in os.listdir(tmp_dir):
        ex.add_artifact(os.path.join(tmp_dir, f))
