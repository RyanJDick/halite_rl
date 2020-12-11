
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
- Currently struggling with Ship Convert Recall metric. Maybe add boolean variable to indicate when it is the last step (if most of the converts are in fact being missed in the last step)
- Update metrics to also only use ship/shipyard locations
- Try smaller batch sizes
- Consider relative weighting of ship and shipyard losses. Maybe focus on just 1 for now.
- metric reporting should count number of gradient updates (i.e. number of batches) rather than number of examples seen.
- think about writing a simple agent to generate data
- Update instructions here
- Add model checkpointing, and resuming
- Weigh loss function more heavily on squares with a ship/shipyard
- Only include squares with ship/shipyard in loss / metrics?
- Add ability to step through examples from data loader in visualizer. This will provide confidence that data loading / conversion is being done properly.

## Training Log

- imitation-cnn_06-Dec-2020_00-47-36
    - SHIPYARD_LOSS_WEIGHT: 3.0, STATE_VALUE_LOSS_WEIGHT: 0.000000001
    - state_value began overfitting quickly, and was still making progress on train set after a full night of training.
    - Loss_ship_action/val was still making steady progress after a night of training
    - Overall Loss was overfitting and trending up on val set.

- imitation-cnn_06-Dec-2020_09-55-23
    - New CNN architecture with residual state value prediction.
    - conv kernel 21x21 to see entire board
    - residual prediction seemed to help value prediction (train loss fell much faster), but this just means that it overfit way faster.