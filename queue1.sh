# Cross val
python -m src.prod.find_lr_and_cross_val with seed=972310501 ground_truth=data/selected/custom_criteria.complete.txt model=scoring.dropout.5-diversity.64
python -m src.prod.find_lr_and_cross_val with seed=972310501 ground_truth=data/selected/custom_criteria.complete.txt model=scoring.dropout.5-diversity.128
python -m src.prod.find_lr_and_cross_val with seed=972310501 ground_truth=data/selected/custom_criteria.complete.txt model=scoring
python -m src.prod.find_lr_and_cross_val with seed=972310501 ground_truth=data/selected/custom_criteria.complete.txt model=scoring.dropout.5

python -m src.prod.find_lr_and_cross_val with seed=972310501 ground_truth=data/selected/fisi_pro_criteria.complete.txt model=scoring.dropout.5-diversity.64
python -m src.prod.find_lr_and_cross_val with seed=972310501 ground_truth=data/selected/fisi_pro_criteria.complete.txt model=scoring.dropout.5-diversity.128
python -m src.prod.find_lr_and_cross_val with seed=972310501 ground_truth=data/selected/fisi_pro_criteria.complete.txt model=scoring
python -m src.prod.find_lr_and_cross_val with seed=972310501 ground_truth=data/selected/fisi_pro_criteria.complete.txt model=scoring.dropout.5

rm -rf /tmp
