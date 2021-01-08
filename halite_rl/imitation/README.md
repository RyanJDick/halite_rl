
## Imitation Training
```
python scripts/download_episodes.py --submission-id 17327012
python scripts/build_datasets.py scripts/imitation_config.yaml
pip install -e .
python scripts/train_imitation.py scripts/imitation_config.yaml
```
