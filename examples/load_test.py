from pprint import pprint

import torch
from huggingface_hub import HfApi

import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


print("List of available datasets:")
pprint(lerobot.available_datasets)

repo_id = "lerobot/aloha_mobile_cabinet"
dataset = LeRobotDataset(repo_id, episodes=[0, 10, 11, 23])

# And see how many frames you have:
print(f"Selected episodes: {dataset.episodes}")
print(f"Number of episodes selected: {dataset.num_episodes}")
print(f"Number of frames selected: {dataset.num_frames}")
print(dataset.meta)
print(dataset.hf_dataset)

# frame indices associated to the first episode:
episode_index = 0
from_idx = dataset.episode_data_index["from"][episode_index].item()
to_idx = dataset.episode_data_index["to"][episode_index].item()


camera_key = dataset.meta.camera_keys[0]
frames = [dataset[idx][camera_key] for idx in range(from_idx, to_idx)]

print(type(frames[0]))
print(frames[0].shape)
pprint(dataset.features[camera_key])
# In particular:
print(dataset.features[camera_key]["shape"])
# The shape is in (h, w, c) which is a more universal format.

dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=0,
    batch_size=1,
    shuffle=True,
)

for batch in dataloader:
    print(f"{batch[camera_key].shape=}")  # (32, 4, c, h, w)
    print(f"{batch['observation.state'].shape=}")  # (32, 6, c)
    print(f"{batch['action'].shape=}")  # (32, 64, c)
    print("| KEY | TYPE | SHAPE |")
    print("| --- | --- | --- |")
    for k, v in batch.items():
        if isinstance(v, list):
            print(f"| {k} | {len(v)} | str | {v} |")
        else:
            print(f"| {k} | {v.dtype} | {v.shape} |")
    import pdb; pdb.set_trace()
