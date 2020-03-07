python -m src.prod.find_lr_and_cross_val with directory=runs/extract_c3d_fc6/3 ground_truth=data/gt/fisi.train.csv model=scoring-dropout.5-diversity.8-nn_grus
python -m src.prod.find_lr_and_cross_val with directory=runs/extract_c3d_fc6/3 ground_truth=data/gt/custom.train.csv model=scoring-dropout.5-diversity.8-nn_grus

python -m src.prod.find_lr_and_cross_val with k_cross_val=1 directory=runs/extract_c3d_fc6/3 ground_truth=data/gt/fisi.train.csv model=scoring-dropout.5-diversity.16-nn_grus
python -m src.prod.find_lr_and_cross_val with k_cross_val=1 directory=runs/extract_c3d_fc6/3 ground_truth=data/gt/custom.train.csv model=scoring-dropout.5-diversity.16-nn_grus

rm -rf /tmp
