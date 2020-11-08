
```
python imitation/download_episodes.py --submission-id 17327012
python imitation/build_pt_dataset.py --submission-id 17327012 --team-name "Stanley Zheng"
```

TODO:
- Create github repo
- Update instructions here
- Add tensorboard
- Split validation set
- Add per-class accuracy metrics
- Add model checkpointing, and resuming
- Get working on GPU
- 
- Add ability to step through examples from data loader in visualizer. This will provide confidence that data loading / conversion is being done properly.
- Add augmentations. Namely: shift/wrap in both x and y directions