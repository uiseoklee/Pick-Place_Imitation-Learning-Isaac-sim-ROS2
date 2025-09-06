# Pick-Place Imitation Learning with Isaac Sim & ROS 2

This project implements Pick-and-Place imitation learning using NVIDIA Isaac Sim and ROS 2.

## Demo Video

![Pick-Place Demo](media/pickplace_imitationlearning_3.gif)

## Technologies Used

- Isaac Sim
- ROS 2
- Python
- Imitation Learning
## What This Repository Enables
- Collect demonstrations in simulated environments
- Train and evaluate a Diffusion Policy mode
- Support the Lite 6 robot arm

## Setup Instructions
### Docker Installation
You need to have Docker installed. If you have an NVIDIA GPU, follow this guide for GPU support.
Isaac Sim must also be installed if you plan to use simulation.
```
sudo apt install git make curl
curl -sSL https://get.docker.com | sh && sudo usermod -aG docker $USER
```
### Clone and Build
```
git clone https://github.com/uiseoklee/Pick-Place_Imitation-Learning-Isaac-sim-ROS2.git
cd Pick-Place_Imitation-Learning-Isaac-sim-ROS2/docker
make build-pc run exec
```
### Build ROS 2 Packages
```
colcon build --symlink-install
source ./install/local_setup.bash
```
## Running Simulation
### Launch ROS 2 Controller
```
ros2 launch xarm_bringup lite6_cartesian_launch.py rviz:=false sim:=true
```
### Run Model in Docker
Open another terminal and run:
```
make exec
cd src/robo_imitate
./imitation/pickplace_redblock
```
### Model Training
Inside the robo_imitate directory
```
docker build --build-arg UID=$(id -u) -t imitation .
docker run -v $(pwd)/imitation/:/docker/app/imitation:Z --gpus all -it \
  -e DATA_PATH=imitation/data/sim_imitation_training_data.parquet \
  -e EPOCH=1000 imitation
```
## Acknowledgments
This repository is based on MarijaGolubovic/robo_imitate.
Special thanks to:
- Marija Golubovic
- @SpesRobotics.
- LeRobot team for open-sourcing LeRobot projects
- Cheng Chi, Zhenjia Xu, and colleagues for open-sourcing Diffusion Policy
