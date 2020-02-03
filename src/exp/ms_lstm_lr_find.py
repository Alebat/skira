from time import time

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
    ground_truth = "data/selected/gt.txt"
    model = 'scoring'


@ex.automain
def main(model, directory, ground_truth, seed):
    random.seed(seed)
    assert os.path.exists(ground_truth), "ground_truth"
    tmp_tr = f"/tmp/train_dataset_{time()}.txt"
    with open(ground_truth, "r") as annotations:
        train = open(tmp_tr, 'w')
        lines = list(annotations)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            name, mark = line.strip().split(',')
            print('c3d-_output-' + '-'.join(name.split('-')[1:]), mark, sep=',', file=train)
        train.close()
    ex.add_artifact(tmp_tr, "train.txt")

    suffixes = [
        ".npy",
        ".mirror.npy",
        ".lfps.npy",
        ".mirror.lfps.npy",
    ]

    trainset = videoDataset(root=directory,
                            label=tmp_tr, suffixes=suffixes, transform=transform, data=None)

    save_file = f'/tmp/plot{time()}.pdf'

    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    # build the model
    scoring = get_scoring_model(model)
    if torch.cuda.is_available():
        scoring.cuda()

    total_params = sum(p.numel() for p in scoring.parameters() if p.requires_grad)
    print("Total Params: " + str(total_params))
    optimizer = optim.Adam(params=scoring.parameters())  # use SGD optimizer to optimize the loss function

    class PLoss(torch.nn.Module):
        def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            logits, penal = input
            if penal is None:
                return scoring.loss(logits, target)
            else:
                return scoring.loss(logits, target) + penal

    lr_finder = LRFinder(scoring, optimizer, PLoss(), device="cuda")
    lr_finder.range_test(trainLoader, end_lr=100, num_iter=100)
    lr_finder.plot().savefig(save_file)

    ex.add_artifact(save_file, "plot.pdf")
