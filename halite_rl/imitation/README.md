
## Imitation Training
```
python scripts/download_episodes.py --submission-id 17327012
python scripts/build_datasets.py scripts/imitation_config.yaml
pip install -e .
python scripts/train_imitation.py scripts/imitation_config.yaml
```

## Unit Tests
```
pip install -e .
pytest tests/
```

TODO:
- Update instructions here
- Add model checkpointing, and resuming
- 
- Add ability to step through examples from data loader in visualizer. This will provide confidence that data loading / conversion is being done properly.
- Add augmentations. Namely: shift/wrap in both x and y directions
