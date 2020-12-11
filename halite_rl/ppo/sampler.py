from halite_rl.utils import SubProcessWrapper
from halite_rl.ppo.halite_env_wrapper import HaliteEnvWrapper


class EpisodeData():
    def __init__():
        self.observations = []  # Observations (states).
        self.actions = []       # Selected actions.
        self.act_log_probs = [] # Log probability of selected action.
        self.value_preds = []   # Value predictions given observation (from critic network).
        self.rewards = []       # Rewards obtained in each step.


def sample_batch(models, env_constructor, min_batch_size, num_parallel_envs=1):
    """
    Sample a batch of environment rollouts.

    Parameters:
    -----------
    models : dict[str: nn.Module]
        Dict mapping player_ids to actor-critic NN models.
    env_constructor : func
        Constructor for environment. Will be passed to SubProcessWrapper if num_parallel_envs > 1.
    num_steps : int
        Number of environment steps to collect.
        (TODO: clarify handling if full episode rollout goes over this)
    num_parallel_envs : int
        Number of environments to run in parallel. Observations from all environments will be
        batched and passed to models together.

    Returns:
    --------

    """

    # Initialize envs.
    envs = [SubProcessWrapper(HaliteEnvWrapper) for _ in range(num_parallel_envs)]
    # EpisodeData for in-progress episodes.
    ep_datas = [EpisodeData() for _ in envs]

    num_steps = 0
    final_ep_datas = []
    while num_steps < min_batch_size:

        # Step all envs.
        # If env not in-progress, check if we have reached min_batch_size and if not reset env.
        # Record rewards(from previous step if we did not just reset), and observation.
        # If episode finished, add to ep_datas and sit this one out.

        # Run actor-critic model.
        # Update actions, act_log_probs, value_preds.

        # Run all envs in parallel.
        for env in envs:


