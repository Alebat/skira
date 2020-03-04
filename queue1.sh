python -m src.prod.find_lr_and_cross_val with directory=runs/extract_c3d_fc6/3 ground_truth=data/gt/fisi.train.csv model=scoring-dropout.5-diversity.16
python -m src.prod.find_lr_and_cross_val with directory=runs/extract_c3d_fc6/3 ground_truth=data/gt/fisi.train.csv model=scoring-dropout.5-diversity.16-nn_lstms
python -m src.prod.find_lr_and_cross_val with directory=runs/extract_c3d_fc6/3 ground_truth=data/gt/fisi.train.csv model=scoring-dropout.5-diversity.16-nn_grus
python -m src.prod.find_lr_and_cross_val with directory=runs/extract_c3d_fc6/3 ground_truth=data/gt/fisi.train.csv model=scoring-dropout.5-skip_grus
python -m src.prod.find_lr_and_cross_val with directory=runs/extract_c3d_fc6/3 ground_truth=data/gt/fisi.train.csv model=scoring-dropout.5-diversity.32
python -m src.prod.find_lr_and_cross_val with directory=runs/extract_c3d_fc6/3 ground_truth=data/gt/fisi.train.csv model=scoring-dropout.5-diversity.64
python -m src.prod.find_lr_and_cross_val with directory=runs/extract_c3d_fc6/3 ground_truth=data/gt/fisi.train.csv model=scoring-dropout.5
python -m src.prod.find_lr_and_cross_val with directory=runs/extract_c3d_fc6/3 ground_truth=data/gt/fisi.train.csv model=scoring

python -m src.prod.find_lr_and_cross_val with directory=runs/extract_c3d_fc6/3 ground_truth=data/gt/custom.train.csv model=scoring-dropout.5-diversity.16
python -m src.prod.find_lr_and_cross_val with directory=runs/extract_c3d_fc6/3 ground_truth=data/gt/custom.train.csv model=scoring-dropout.5-diversity.16-nn_lstms
python -m src.prod.find_lr_and_cross_val with directory=runs/extract_c3d_fc6/3 ground_truth=data/gt/custom.train.csv model=scoring-dropout.5-diversity.16-nn_grus
python -m src.prod.find_lr_and_cross_val with directory=runs/extract_c3d_fc6/3 ground_truth=data/gt/custom.train.csv model=scoring-dropout.5-skip_grus
python -m src.prod.find_lr_and_cross_val with directory=runs/extract_c3d_fc6/3 ground_truth=data/gt/custom.train.csv model=scoring-dropout.5-diversity.32
python -m src.prod.find_lr_and_cross_val with directory=runs/extract_c3d_fc6/3 ground_truth=data/gt/custom.train.csv model=scoring-dropout.5-diversity.64
python -m src.prod.find_lr_and_cross_val with directory=runs/extract_c3d_fc6/3 ground_truth=data/gt/custom.train.csv model=scoring-dropout.5
python -m src.prod.find_lr_and_cross_val with directory=runs/extract_c3d_fc6/3 ground_truth=data/gt/custom.train.csv model=scoring

rm -rf /tmp
