import torch
import datasets
from torch import Tensor, nn
from safetensors.torch import load_file
import numpy as np


def populate_queues(queues, batch):
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        if key not in queues:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues


def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """Get a module's parameter dtype by checking one of its parameters.

    Note: assumes that all parameters have the same dtype.
    """
    return next(iter(module.parameters())).dtype


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Note: assumes that all parameters have the same device
    """
    return next(iter(module.parameters())).device


# used
def create_stats_buffers(
    shapes: dict[str, list[int]],
    modes: dict[str, str],
    stats: dict[str, dict[str, Tensor]] | None = None,
) -> dict[str, dict[str, nn.ParameterDict]]:
    """
    Create buffers per modality (e.g. "observation.image", "action") containing their mean, std, min, max
    statistics.

    Args: (see Normalize and Unnormalize)

    Returns:
        dict: A dictionary where keys are modalities and values are `nn.ParameterDict` containing
            `nn.Parameters` set to `requires_grad=False`, suitable to not be updated during backpropagation.
    """
    stats_buffers = {}

    for key, mode in modes.items():
        assert mode in ["mean_std", "min_max"]

        shape = tuple(shapes[key])

        if "image" in key:
            # sanity checks
            assert len(shape) == 3, f"number of dimensions of {key} != 3 ({shape=}"
            c, h, w = shape
            assert c < h and c < w, f"{key} is not channel first ({shape=})"
            # override image shape to be invariant to height and width
            shape = (c, 1, 1)

        # Note: we initialize mean, std, min, max to infinity. They should be overwritten
        # downstream by `stats` or `policy.load_state_dict`, as expected. During forward,
        # we assert they are not infinity anymore.

        buffer = {}
        if mode == "mean_std":
            mean = torch.ones(shape, dtype=torch.float32) * torch.inf
            std = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "mean": nn.Parameter(mean, requires_grad=False),
                    "std": nn.Parameter(std, requires_grad=False),
                }
            )
        elif mode == "min_max":
            min = torch.ones(shape, dtype=torch.float32) * torch.inf
            max = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "min": nn.Parameter(min, requires_grad=False),
                    "max": nn.Parameter(max, requires_grad=False),
                }
            )

        if stats is not None:
            # Note: The clone is needed to make sure that the logic in save_pretrained doesn't see duplicated
            # tensors anywhere (for example, when we use the same stats for normalization and
            # unnormalization). See the logic here
            # https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/py_src/safetensors/torch.py#L97.
            if mode == "mean_std":
                buffer["mean"].data = stats[key]["mean"].clone()
                buffer["std"].data = stats[key]["std"].clone()
            elif mode == "min_max":
                buffer["min"].data = stats[key]["min"].clone()
                buffer["max"].data = stats[key]["max"].clone()

        stats_buffers[key] = buffer
    return stats_buffers


# def _no_stats_error_str(name: str) -> str:
#     return (
#         f"`{name}` is infinity. You should either initialize with `stats` as an argument, or use a "
#         "pretrained model."
#     )

def unflatten_dict(d, sep="/"):
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict


def flatten_dict(d, parent_key="", sep="/"):
    """Flatten a nested dictionary structure by collapsing nested keys into one key with a separator.

    For example:
    ```
    >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}`
    >>> print(flatten_dict(dct))
    {"a/b": 1, "a/c/d": 2, "e": 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)



def load_stats(repo_id, version, root) -> dict[str, dict[str, torch.Tensor]]:
    """stats contains the statistics per modality computed over the full dataset, such as max, min, mean, std

    Example:
    ```python
    normalized_action = (action - stats["action"]["mean"]) / stats["action"]["std"]
    ```
    """
    
    stats = load_file(root)
    return unflatten_dict(stats)


def load_previous_and_future_frames(
    item: dict[str, torch.Tensor],
    hf_dataset: datasets.Dataset,
    episode_data_index: dict[str, torch.Tensor],
    delta_timestamps: dict[str, list[float]],
    tolerance_s: float,
) -> dict[torch.Tensor]:
    """
    Given a current item in the dataset containing a timestamp (e.g. 0.6 seconds), and a list of time differences of
    some modalities (e.g. delta_timestamps={"observation.image": [-0.8, -0.2, 0, 0.2]}), this function computes for each
    given modality (e.g. "observation.image") a list of query timestamps (e.g. [-0.2, 0.4, 0.6, 0.8]) and loads the closest
    frames in the dataset.

    Importantly, when no frame can be found around a query timestamp within a specified tolerance window, this function
    raises an AssertionError. When a timestamp is queried before the first available timestamp of the episode or after
    the last available timestamp, the violation of the tolerance doesnt raise an AssertionError, and the function
    populates a boolean array indicating which frames are outside of the episode range. For instance, this boolean array
    is useful during batched training to not supervise actions associated to timestamps coming after the end of the
    episode, or to pad the observations in a specific way. Note that by default the observation frames before the start
    of the episode are the same as the first frame of the episode.

    Parameters:
    - item (dict): A dictionary containing all the data related to a frame. It is the result of `dataset[idx]`. Each key
      corresponds to a different modality (e.g., "timestamp", "observation.image", "action").
    - hf_dataset (datasets.Dataset): A dictionary containing the full dataset. Each key corresponds to a different
      modality (e.g., "timestamp", "observation.image", "action").
    - episode_data_index (dict): A dictionary containing two keys ("from" and "to") associated to dataset indices.
      They indicate the start index and end index of each episode in the dataset.
    - delta_timestamps (dict): A dictionary containing lists of delta timestamps for each possible modality to be
      retrieved. These deltas are added to the item timestamp to form the query timestamps.
    - tolerance_s (float, optional): The tolerance level (in seconds) used to determine if a data point is close enough to the query
      timestamp by asserting `tol > difference`. It is suggested to set `tol` to a smaller value than the
      smallest expected inter-frame period, but large enough to account for jitter.

    Returns:
    - The same item with the queried frames for each modality specified in delta_timestamps, with an additional key for
      each modality (e.g. "observation.image_is_pad").

    Raises:
    - AssertionError: If any of the frames unexpectedly violate the tolerance level. This could indicate synchronization
      issues with timestamps during data collection.
    """
    # get indices of the frames associated to the episode, and their timestamps
    ep_id = item["episode_index"].item()
    ep_data_id_from = episode_data_index["from"][ep_id].item()
    ep_data_id_to = episode_data_index["to"][ep_id].item()
    
    ep_data_ids = torch.arange(ep_data_id_from, ep_data_id_to, 1) 
    ep_timestamps = hf_dataset.select_columns("timestamp")[ep_data_id_from:ep_data_id_to]["timestamp"]
    
    
    # list elements to torch, because stack require tuple of tensors
    ep_timestamps = [torch.tensor(item) for item in ep_timestamps]
    ep_timestamps = torch.stack(ep_timestamps)
    
    # we make the assumption that the timestamps are sorted
    ep_first_ts = ep_timestamps[0]
    ep_last_ts = ep_timestamps[-1]
    current_ts = item["timestamp"].item()
    
    for key in delta_timestamps:
        # get timestamps used as query to retrieve data of previous/future frames
        delta_ts = delta_timestamps[key]
        query_ts = current_ts + torch.tensor(delta_ts)
        

        # compute distances between each query timestamp and all timestamps of all the frames belonging to the episode
        dist = torch.cdist(query_ts[:, None], ep_timestamps[:, None], p=1)
        min_, argmin_ = dist.min(1)

        # TODO(rcadene): synchronize timestamps + interpolation if needed

        is_pad = min_ > tolerance_s

        # check violated query timestamps are all outside the episode range
        assert ((query_ts[is_pad] < ep_first_ts) | (ep_last_ts < query_ts[is_pad])).all(), (
            f"One or several timestamps unexpectedly violate the tolerance ({min_} > {tolerance_s=}) inside episode range."
            "This might be due to synchronization issues with timestamps during data collection."
        )

        # get dataset indices corresponding to frames to be loaded
        data_ids = ep_data_ids[argmin_]
    
        # load frames modality
        item[key] = hf_dataset.select_columns(key).select(data_ids)[key]
        column = hf_dataset.select_columns(key).select(data_ids)
        
        # get element from sorted list
        # function from lib take only one element in original implementation work
        items= []
        for i in column:
            items.append(i[key])
        item[key] = items.copy()

        # list elements to torch, because stack require tuple of tensors
        item[key] = [torch.tensor(item) for item in item[key]]
        item[key] = torch.stack(item[key])

        item[f"{key}_is_pad"] = is_pad
    
    return item


def plot_action_trajectory(sim, positions1, positions2=None):
    import matplotlib.pyplot as plt
    from datetime import datetime
    import os

    positions1 = np.array(positions1)
    norm1 = plt.Normalize(positions1[:, 2].min(), positions1[:, 2].max())
    print(norm1)
    colors1 = plt.cm.viridis(norm1(positions1[:, 2]))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc1 = ax.scatter(positions1[:, 0], positions1[:, 1], positions1[:, 2], c=colors1, marker='o', s=50, alpha=0.8, edgecolor='k', linewidth=0.5, label='Trajektorija observacije')

    if positions2 is not None:
        positions2 = np.array(positions2)
        norm2 = plt.Normalize(positions2[:, 2].min(), positions2[:, 2].max())
        colors2 = plt.cm.plasma(norm2(positions2[:, 2]))
        print(norm2)

        sc2 = ax.scatter(positions2[:, 0], positions2[:, 1], positions2[:, 2], c=colors2, marker='^', s=50, alpha=0.8, edgecolor='k', linewidth=0.5, label='Trajektorija akcije')

    ax.set_xlabel('X osa', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y osa', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z osa', fontsize=12, fontweight='bold')

    ax.set_title('Razlika između akcije i observacije', fontsize=14, fontweight='bold')

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.legend(loc='best')

    now = datetime.now()
    real_path = 'action_trajectoris/real/'
    sim_path = 'action_trajectoris/sim/'

    if not os.path.exists('action_trajectoris'):
        os.mkdir('action_trajectoris')
    if not os.path.exists(real_path):
        os.mkdir(real_path)
    if not os.path.exists(sim_path):
        os.mkdir(sim_path)

    name = real_path + now.strftime("%Y_%m_%d_%H_%M_%S") + '.png'
    if sim:
        name = sim_path + now.strftime("%Y_%m_%d_%H_%M_%S") + '.png'

    plt.savefig(name, bbox_inches='tight', dpi=300)
    