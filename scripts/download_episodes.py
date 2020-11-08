import argparse
import os
import requests
import json


def get_episodes_list(submission_id):
    list_url = "https://www.kaggle.com/requests/EpisodeService/ListEpisodes"
    #resp = requests.post(list_url, json={"teamId": team_id})
    resp = requests.post(list_url, json={"SubmissionId": submission_id})

    return resp.json()['result']['episodes']

def download_episode(episode_id, dir):
    episode_replay_url = "https://www.kaggle.com/requests/EpisodeService/GetEpisodeReplay"

    out_path = os.path.join(dir, f"{episode_id}.json")
    if os.path.exists(out_path):
        print(f"'{out_path}' already exists. Skipping.")
        return

    try:
        resp = requests.post(episode_replay_url, json={"EpisodeId": int(episode_id)})
    except Exception as e:
        print(f"Failed to download episode '{episode_id}': {str(e)}")
        return

    resp = resp.json()
    if not resp['wasSuccessful']:
        print(f"Episode '{episode_id}' wasSuccesful != True. Skipping.")
        return

    os.makedirs(dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(resp['result'], f)
    print(f"Downloaded '{out_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-id", help="submission_id for which to download episode")
    args = parser.parse_args()

    episodes = get_episodes_list(args.submission_id)
    for i, ep in enumerate(episodes):
        if i % 100 == 0:
            print(f"Downloading episode {i+1} / {len(episodes)}")
        download_episode(ep['id'], f"./data/submission_{args.submission_id}_episodes/")
    print("Done.")
