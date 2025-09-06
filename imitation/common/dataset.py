import torch
import datasets
from pathlib import Path
from typing import Dict
from datasets import load_dataset
from PIL import Image as PILImage
import os
from torchvision import transforms
from common.utils import load_stats, load_previous_and_future_frames



DATA_DIR = Path(os.environ["DATA_DIR"]) if "DATA_DIR" in os.environ else None
CODEBASE_VERSION = "v1.4"

def hf_transform_to_torch(items_dict):
    from PIL import Image
    import io
    for key in items_dict:
        first_item = items_dict[key][0]
        
        if key == 'observation.image':
            first_item = items_dict[key][0]['bytes']
            # print(first_item)
            # print('------------')
            first_item = Image.open(io.BytesIO(first_item))

           
        if isinstance(first_item, PILImage.Image):
            to_tensor = transforms.ToTensor()
            for item in items_dict[key]:
                
                item = item['bytes']
                items_dict[key] = to_tensor(Image.open(io.BytesIO(item)))
                items_dict[key] = items_dict[key].unsqueeze(0)
                
                
        elif key == 'observation.state' or key == 'action':
                import numpy as np
                for next_line  in items_dict[key]:
                    next_line = next_line[1:-1].split(',')
                    next_line = [float(item) for item in next_line]
                    next_line = np.array(next_line)

                    items_dict[key] = torch.tensor(next_line, dtype=torch.float32)
                    items_dict[key] = items_dict[key].unsqueeze(0)
        else:

            items_dict[key] = torch.tensor(items_dict[key])
            
    return items_dict


def load_hf_dataset(repo_id, version, root, split) -> datasets.Dataset:
    
    if root is not None:
        print("roooooooooooooooot: ", root)
        hf_dataset = load_dataset('parquet', data_files = root, split = split)
    print()
    hf_dataset.set_transform(hf_transform_to_torch)

    return hf_dataset




def calculate_episode_data_index_for_custom_dataset(hf_dataset: datasets.Dataset) -> Dict[str, torch.Tensor]:
    episode_data_index = {"from": [], "to": []}

    current_episode = None

    if len(hf_dataset) == 0:
        episode_data_index = {
            "from": torch.tensor([]),
            "to": torch.tensor([]),
        }
        return episode_data_index

    for idx, episode_idx in enumerate(hf_dataset["episode_index"]):
        if episode_idx != current_episode:
            # We encountered a new episode, so we append its starting location to the "from" list
            episode_data_index["from"].append(idx)
            # If this is not the first episode, we append the ending location of the previous episode to the "to" list
            if current_episode is not None:
                episode_data_index["to"].append(idx)
            # Let's keep track of the current episode index
            current_episode = episode_idx

    # Add the ending index for the last episode
    episode_data_index["to"].append(idx + 1)

    # Convert lists to tensors
    episode_data_index["from"] = torch.tensor(episode_data_index["from"])
    episode_data_index["to"] = torch.tensor(episode_data_index["to"])

    return episode_data_index


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        version: str | None = CODEBASE_VERSION,
        root: Path | None = DATA_DIR,
        split: str = "train",
        transform: callable = None,
        delta_timestamps: dict[list[float]] | None = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.version = version
        self.root = root
        self.split = split
        self.transform = transform
        self.delta_timestamps = delta_timestamps
      
        self.hf_dataset = load_hf_dataset(repo_id, version, root, split)
        if split == "train":
            self.episode_data_index = calculate_episode_data_index_for_custom_dataset(self.hf_dataset)
        else:
            self.episode_data_index = calculate_episode_data_index_for_custom_dataset(self.hf_dataset)
        
        # safe_tensors_path = 'This need add path to stats path'
        safe_tensors_path = 'imitation/metadata/stats.safetensors'
        self.stats = load_stats(repo_id, version, root=safe_tensors_path)

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return 10

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.
        Returns False if it only loads images from png files.
        """
        return None

    @property
    def features(self) -> datasets.Features:
        return self.hf_dataset.features

    @property
    def num_samples(self) -> int:
        """Number of samples/frames."""
        return len(self.hf_dataset)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return len(self.hf_dataset.unique("episode_index"))

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        if self.delta_timestamps is not None:
            item = load_previous_and_future_frames(
                item,
                self.hf_dataset,
                self.episode_data_index,
                self.delta_timestamps,
                self.tolerance_s,
            )

        if self.transform is not None:
            item = self.transform(item)

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository ID: '{self.repo_id}',\n"
            f"  Version: '{self.version}',\n"
            f"  Split: '{self.split}',\n"
            f"  Number of Samples: {self.num_samples},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            # f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.transform},\n"
            f")"
        )

    @classmethod
    def from_preloaded(
        cls,
        repo_id: str,
        version: str | None = CODEBASE_VERSION,
        root: Path | None = None,
        split: str = "train",
        transform: callable = None,
        delta_timestamps: dict[list[float]] | None = None,
        # additional preloaded attributes
        hf_dataset=None,
        episode_data_index=None,
        stats=None
    ):
        # create an empty object of type LeRobotDataset
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.version = version
        obj.root = root
        obj.split = split
        obj.transform = transform
        obj.delta_timestamps = delta_timestamps
        obj.hf_dataset = hf_dataset
        obj.episode_data_index = episode_data_index,
        obj.stats = stats
        return obj
