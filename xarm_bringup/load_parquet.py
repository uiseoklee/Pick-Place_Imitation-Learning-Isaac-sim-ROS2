import pandas as pd
import numpy as np
import cv2

# file_path = '/home/marija/Downloads/train-00000-of-00001.parquet'
file_path = '/home/marija/spes_autonomy/xarm_bringup/DATA/parquest_output/2024_08_20_13_40_11.parquet'
df = pd.read_parquet(file_path, engine='pyarrow')

for index, row in df.iterrows():
    observation_image = row['observation.image']
    
    observation_state = row['observation.state']
    action = row['action']
    episode_index = row['episode_index']
    frame_index = row['frame_index']
    timestamp = row['timestamp']
    next_reward = row['next.reward']
    next_done = row['next.done']
    next_success = row['next.success']
    index_value = row['index']


    print(f"Row {index}:")
    print("Observation State:", observation_state)
    print("Action:", action)
    print("Episode Index:", episode_index)
    print("Frame Index:", frame_index)
    print("Timestamp:", timestamp)
    print("Next Reward:", next_reward)
    print("Next Done:", next_done)
    print("Next Success:", next_success)
    print("Index:", index_value)
    print("\n")

    image_array = np.frombuffer(observation_image['bytes'], dtype=np.uint8)
    print('image size: ', image_array.shape)

    
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # cv2.imshow('image', image)
    if episode_index == 1:
        break
        # cv2.waitKey(0)
        # print("---------------------------------------------------------------------------------------------------------------------")
    # cv2.waitKey(0)

