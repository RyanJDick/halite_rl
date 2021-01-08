## TODO:

* Current priority: settle on network arch that gives good results for both action and value prediction.
* Train PPO from imitation model.
* improve value prediction by including more skip connections from inputs to final layers
* look into gradient clipping
* Shoudn't be using root user in docker container
* Profiling to find bottlenecks and speed up training. I am under the impression that the halite environment implementation is really slow.
* Set up self-play framework.
    * Read about how this was done in AlphaZero and other works.
    * Support both players learning at the same time? Or have one frozen?
    * Do I need to maintain a league of different agents? Or can I just have one agent playing against itself?
    * Single agent would probably be preferred given single GPU (and have to have a critic model on there as well)
