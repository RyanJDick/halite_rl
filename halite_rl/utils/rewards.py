

def calc_episode_reward(cur_team_halite_reward, other_team_halite_reward):
    """Episode reward function.

    The current implementation is very simple: the difference between the current team's Halite
    reward and the opponents Halite reward.
    (Note: the Halite reward is assigned a negative value for teams that get eliminated early.)
    """
    return cur_team_halite_reward - other_team_halite_reward
