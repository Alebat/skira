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


@ex.automain
def main(directory, ground_truth, test_set, seed, model):
    random.seed(seed)
    assert os.path.exists(ground_truth), "ground_truth"

    testset = videoDataset(root=directory,
                           label=test_set, suffixes='.npy', transform=transform, data=None)

    testLoader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

    # build the model
    scoring = Scoring(feature_size=4096)
    if torch.cuda.is_available():
        scoring.cuda()

    scoring.load_state_dict(torch.load(model))
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
    print('truth', 'pred', sep=', ')
    for i in range(val_truth.shape[0]):
        print(str(val_truth[i]), str(val_pred[i]), sep=', ')
