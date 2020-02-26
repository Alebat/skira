from time import time

import numpy as np
import os
import pandas as pd
import random
import torch
import torch.utils.data as data
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, pearsonr
from torch.autograd import Variable

from MS_LSTM.dataloader import videoDataset, transform
from MS_LSTM.model import get_scoring_model
from src.exp.base_experiment import get_skira_exp

ex = get_skira_exp("test_ms_lstm")


@ex.config
def config_1():
    test_size = None
    ground_truth = None
    epochs = None
    lr_range = None
    text_filter = ''


@ex.automain
def main(directory, test_set, seed, model, weights, text_filter):
    random.seed(seed)

    suffixes = list(filter(lambda x: text_filter in x, [
        ".npy",
        ".flip.npy",
        ".lfps.npy",
        ".flip.lfps.npy",
    ]))

    testset = videoDataset(root=directory,
                           label=test_set, suffixes=suffixes[0], transform=lambda x, _: transform(x, 1000), data=None)

    testLoader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

    # build the model
    scoring = get_scoring_model(model)
    scoring.load_state_dict(torch.load(weights))
    if torch.cuda.is_available():
        scoring.cuda()

    with torch.no_grad():
        scoring.eval()
        val_pred = []
        val_sample = 0
        val_loss = 0
        val_truth = []
        for features, scores in testLoader:
            val_truth.append(scores.numpy())
            if torch.cuda.is_available():
                features = Variable(features).cuda()
                scores = Variable(scores).cuda()
            regression, _ = scoring(features)
            val_pred.append(regression.data.cpu().numpy())
            regr_loss = scoring.loss(regression, scores)
            val_loss += (regr_loss.data.item()) * scores.shape[0]
            val_sample += scores.shape[0]
        val_truth = np.concatenate(val_truth)[:, 0]
        val_pred = np.concatenate(val_pred)[:, 0]

    to_data = f'/tmp/{time()}_ms_lstm_test.csv'
    to_plot = f'/tmp/{time()}_plot.pdf'

    with open(to_data, 'w') as f:
        for i in range(val_truth.shape[0]):
            print(str(val_truth[i]), str(val_pred[i]), sep=',', file=f)

    ex.add_artifact(to_data, 'data.csv')
    os.remove(to_data)

    print('SpearmanCorrelation:', spearmanr(val_truth, val_pred))
    print('PearsonCorrelation:', pearsonr(val_truth, val_pred))
    print('MSE:', (np.square(val_truth - val_pred)).mean())

    plt.scatter(val_truth, val_pred)
    plt.xlabel("Truth")
    plt.ylabel("Predicted")
    plt.savefig(to_plot)
    plt.close()

    ex.add_artifact(to_plot, 'plot_scatter.pdf')

    gb = pd.DataFrame(np.array([val_truth, val_pred]).T).groupby(0)
    gb = np.array([[i[0], list(i[1][1])] for i in gb])
    labels = gb[:, 0]
    gb = gb[:, 1]

    plt.boxplot(gb, labels=labels)
    plt.xlabel("Truth")
    plt.ylabel("Predicted")
    plt.savefig(to_plot)

    ex.add_artifact(to_plot, 'plot_boxplot.pdf')
