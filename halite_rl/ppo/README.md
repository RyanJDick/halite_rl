```
pip install -e .
python scripts/train_ppo.py scripts/ppo_config.yaml
```

## Training Log:

- `actor-critic-cnn_24-Dec-2020_14-46-38` was trained against a random agent. It learned to survive long enough that the random agent self-destructed. Not clear whether it learned to evade the opponent's ships.
- 