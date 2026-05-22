# Phase 1 — Linux box setup (NOT in this repo)

This document is a checklist for when you provision a Linux dev box and want
to graduate from the macOS Phase 0 sandbox to a real Gazebo + PX4 + ROS 2 + Crazyflie environment.

## Target machine

- Ubuntu 24.04 LTS.
- 64 GB RAM.
- RTX 3090 / 4090 (24 GB+) for Phase 2 MARL.
- 1 TB NVMe.

## Stack (top → bottom)

```bash
# ROS 2 Jazzy + Gazebo Harmonic
sudo apt update && sudo apt install -y ros-jazzy-desktop ros-jazzy-ros-gz
curl -sSL http://get.gazebosim.org | sh

# PX4 v1.17
git clone --recursive --branch v1.17 https://github.com/PX4/PX4-Autopilot.git
cd PX4-Autopilot && bash ./Tools/setup/ubuntu.sh

# Smoke test
make px4_sitl gz_x500
./Tools/simulation/gz/multi_vehicle.sh -n 10 -m gz_x500

# uXRCE-DDS agent
sudo apt install -y micro-xrce-dds-agent
MicroXRCEAgent udp4 -p 8888

# Crazyswarm2
git clone https://github.com/IMRCLab/crazyswarm2.git ~/ros2_ws/src/crazyswarm2
cd ~/ros2_ws && colcon build --symlink-install

# Zenoh (rmw_zenoh)
sudo apt install -y ros-jazzy-rmw-zenoh-cpp
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
```

## Swap-in code

Replace in this repo (do NOT do this on macOS):

| File | New impl |
|---|---|
| `src/sargvision_swarm/comms/bus.py` | `ZenohBus` using `zenoh-python` |
| `src/sargvision_swarm/sim/simple_sim.py` | `GazeboPX4Sim` calling MAVSDK + uXRCE-DDS |
| new: `src/sargvision_swarm/sim/crazyflie.py` | Crazyswarm2 wrapper |

The `LLMBackend` Protocol + reflex layer stay identical — those are platform-agnostic.

## Phase 1 acceptance test

- 10 drones in Gazebo Harmonic, PX4 SITL, ROS 2 Jazzy, Zenoh.
- `swarm sim --n 10 --scenario formation_v` plus `--backend gazebo` produces convergent V-formation in PX4 telemetry.
- Crazyflie 2.1+ (5-drone fleet) reproduces the same scenario indoors.

## Detail reference

See `~/Documents/AI_Workspace/drone_swarm_research/00_MASTER.md` and the individual reports `01_simulators.md`, `02_autopilots.md`, `04_comms.md`, `03_crazyflie.md`.
