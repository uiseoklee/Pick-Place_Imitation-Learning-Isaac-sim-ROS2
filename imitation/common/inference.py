
import numpy as np
import torch

from common.diffusion_policy import DiffusionPolicy
from pathlib import Path


class Imitation():
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda")
        self.pretrained_policy_path = Path("imitation/outputs/train")

        self.policy = DiffusionPolicy.from_pretrained(self.pretrained_policy_path)
        print(self.policy)
        self.policy.eval()
        self.policy.to(self.device)
        
        self.policy.reset()

    
    def step(self, observation_image, observation_pose):
        # Prepare observation for the policy running in Pytorch
        state = torch.from_numpy(np.array(observation_pose))
        state = state.to(torch.float32).to(self.device, non_blocking=True).unsqueeze(0)
        image = torch.from_numpy(observation_image).to(torch.float32).permute(2, 0, 1).to(self.device, non_blocking=True).unsqueeze(0) / 255

        # Create the policy input dictionary
        observation = {
            "observation.state": state,
            "observation.image": image,
        }

        # Predict the next action with respect to the current observation
        with torch.inference_mode():
            action = self.policy.select_action(observation).squeeze(0).cpu().numpy()
            return action