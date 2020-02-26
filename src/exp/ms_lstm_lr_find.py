from time import time

import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.utils.data as data

from MS_LSTM.dataloader import videoDataset, transform
from MS_LSTM.model import get_scoring_model
from src.exp.base_experiment import get_skira_exp
from src.util.lr_finder import LRFinder

ex = get_skira_exp("lr_find_ms_lstm")


@ex.config
def config_1():
    model = 'scoring'
    no_nfps = False
    no_lfps = False
    no_flip = False
    featn = 4096


@ex.automain
def main(model, directory, ground_truth, seed, no_nfps, no_flip, no_lfps, featn):
    random.seed(seed)
    assert os.path.exists(ground_truth), "ground_truth"

    with open(ground_truth, "r") as annotations:
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

    tmp_tr = f"/tmp/lr_find_dataset_{time()}.txt"
    tmp_pl = f"/tmp/lr_find_plot_{time()}.pdf"

    with open(tmp_tr, 'w') as train:
        for i, line in enumerate(lines):
            name, mark = line.strip().split(',')
            print(prefix + '-'.join(name.split('-')[1:]), mark, sep=',', file=train)

    ex.add_artifact(tmp_tr, f"dataset.txt")

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

    print('Min found in', lrs[np.argmin(losses)])

    lr_finder.plot().savefig(tmp_pl)

    ex.add_artifact(tmp_pl, "plot.pdf")
