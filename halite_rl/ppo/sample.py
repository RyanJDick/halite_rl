import numpy as np

import torch

from halite_rl.utils import SubProcessWrapper


class EpisodeData():
    def __init__(self):
        self.observations = []  # Observations (states).
        self.actions = []       # Selected actions.
        self.act_log_probs = [] # Log probability of selected action.
        self.value_preds = []   # Value predictions given observation (from critic network).
        self.rewards = []       # Rewards obtained in each step.


def sample_batch(models, env_constructor, device, config):
    """Sample a batch of environment rollouts.

    Parameters:
    -----------
    models : dict[str: nn.Module]
        Dict mapping player_ids to actor-critic NN models.
    config : dict
        Config settings.

    Returns:
    --------
    TODO

    """

    # Initialize envs.
    envs = [SubProcessWrapper(env_constructor) for _ in range(config["SAMPLE_PARALLEL_ENVS"])]

    player_ids = list(models.keys())

    # EpisodeData for in-progress episodes.
    # ep_datas[i][p_id] references the EpisodeData for player p_id in the i'th env.
    ep_datas = [{p_id: None for p_id in player_ids} for _ in envs]

    # actions[i][p_id] references the action for player p_id in the i'th env.
    actions = [{p_id: None for p_id in player_ids} for _ in envs]

    num_steps = {p_id: 0 for p_id in player_ids}

    # final_ep_datas[p_id][i] references the EpisodeData for the i'th episode collected for player p_id.
    final_ep_datas = {p_id: [] for p_id in player_ids}

    # While at least one player is below SAMPLE_MIN_NUM_STEPS.
    while np.any(np.array([n for n in num_steps.values()]) < config["SAMPLE_MIN_NUM_STEPS"]):
        # 1. Step all envs asynchronously.

        # Keep a record of which envs were 'reset' and which were 'stepped' so that we
        # know what return values to expect when we receive the results asynchronously.
        env_was_reset = []
        for i_env, env in enumerate(envs):
            if not env.call_sync("is_in_progress"):
                env_was_reset.append(True)
                for p_id in player_ids:
                    ep_data = ep_datas[i_env][p_id]
                    # If this is not the very first iteration, then save the episode.
                    if ep_data is not None:
                        # Drop the last observation, as we never acted on it.
                        ep_data.observations = ep_data.observations[:len(ep_data.rewards)]
                        final_ep_datas[p_id].append(ep_data)
                        num_steps[p_id] += len(ep_data.rewards)
                ep_datas[i_env] = {p_id: EpisodeData() for p_id in player_ids}
                env.call_async("reset")
            else:
                env_was_reset.append(False)
                actions = {p_id: ep_datas[i_env][p_id].actions[-1] for p_id in player_ids}
                env.call_async("step", actions)
        # 2. Receive results from async env steps.

        for i_env, env in enumerate(envs):
            if env_was_reset[i_env]:
                obs = env.get_result()
                for p_id in player_ids:
                    ep_datas[i_env][p_id].observations.append(obs[p_id])
            else:
                obs, rewards, dones = env.get_result()
                for p_id in player_ids:
                    ep_data = ep_datas[i_env][p_id]
                    ep_data.observations.append(obs[p_id])
                    ep_data.rewards.append(rewards[p_id])

        # 3. Sample actions.

        player_id_to_state_batch = {p_id: [] for p_id in player_ids}
        for i_env, env in enumerate(envs):
            for p_id in player_ids:
                player_id_to_state_batch[p_id].append(ep_datas[i_env][p_id].observations[-1])

        for p_id in player_ids:
            model = models[p_id]
            with torch.no_grad():
                state_batch = np.array(player_id_to_state_batch[p_id])
                state_batch = torch.Tensor(state_batch)
                state_batch = state_batch.to(device)
                ship_act_logits, shipyard_act_logits, value_preds = model(state_batch)

                ship_action_dist = model.apply_action_distribution(ship_act_logits)
                shipyard_action_dist = model.apply_action_distribution(shipyard_act_logits)

                ship_action = ship_action_dist.sample()
                shipyard_action = shipyard_action_dist.sample()

                action_log_prob = model.action_log_prob(
                    ship_action_dist,
                    shipyard_action_dist,
                    state_batch,
                    ship_action,
                    shipyard_action,
                )

                ship_action = ship_action.cpu().detach().numpy()
                shipyard_action = shipyard_action.cpu().detach().numpy()
                action_log_prob = action_log_prob.cpu().detach().numpy()
                value_preds = value_preds.cpu().detach().numpy()

            for i_env, env in enumerate(envs):
                if env.call_sync("is_in_progress"):
                    ep_data = ep_datas[i_env][p_id]
                    ep_data.actions.append((
                        ship_action[i_env, ...],
                        shipyard_action[i_env, ...],
                    ))
                    ep_data.act_log_probs.append(action_log_prob[i_env])
                    ep_data.value_preds.append(value_preds[i_env])

    # Close all envs
    for e in envs:
        e.close()

    return final_ep_datas
