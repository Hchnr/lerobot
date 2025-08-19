import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lerobot.policies.factory import make_policy
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "/share/project/hcr/models/lerobot/pi0"
DATASET_REPO_ID = "lerobot/aloha_mobile_cabinet"
DATASET = LeRobotDataset(DATASET_REPO_ID, episodes=[0, 10, 11, 23])
NUM_INFER = 50

def main():
    dataloader = torch.utils.data.DataLoader(
        DATASET,
        num_workers=0,
        batch_size=1,
    )
    batch = next(iter(dataloader))

    # To device
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device=DEVICE, dtype=torch.float32)

    cfg = PreTrainedConfig.from_pretrained(MODEL_PATH)
    cfg.pretrained_path = MODEL_PATH
    policy = make_policy(cfg, ds_meta=DATASET.meta)
    # policy = torch.compile(policy, mode="reduce-overhead")

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens, lang_masks = policy.prepare_language(batch)

    for _ in range(NUM_INFER):
        t_s = time.time()
        actions = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=None)
        print(f"sample_actions: {(time.time() - t_s)*1000:2f} ms")
        original_action_dim = policy.config.action_feature.shape[0]
        actions = actions[:, :10, :original_action_dim]
    print("actions: ", actions.shape)


if __name__ == "__main__":
    main()
