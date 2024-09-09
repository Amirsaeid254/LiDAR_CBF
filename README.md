## High-Order Time-Varying Barrier Functions for Safety in Unmapped and Dynamic Environments

![contour_plot_animation (2)](https://github.com/user-attachments/assets/ccf6fb10-cd6b-4b6f-8846-0bb3963a22c6)

This repository provides implementations of **High-Order Time-Varying Barrier Functions** for safety in unmapped and dynamic environments. It includes classes for LiDAR sensors, synthesizing barrier functions from sensor data, and the time-varying smooth composition of barriers. This package depends on [HOCBF Composition](https://github.com/pedramrabiee/hocbf_composition) to create barrier objects and synthesize safe control inputs. 

### LiDAR Class
The `LiDAR` class returns the sensor data based on the robot's position and the navigating environment. 
- `update(x)`: Returns the sensor data corresponding to the robot's current position.

#### Parameters

The `LiDAR` class takes the following parameters for initialization:

- **max_range**: The maximum range of the LiDAR sensor (e.g., `5.0`).
- **ray_num**: The number of rays emitted by the sensor in each scan (e.g., `100`).
- **scan_angle**: The angular range for the LiDAR scan, defined as `[start_angle, end_angle]` in radians (e.g., `[-π, π]`).
- **update_rate**: The rate at which the sensor data is updated, in seconds (e.g., `0.2`).
- **ray_sampling_rate**: The number of points sampled along each ray (e.g., `1000`).
- **space_dimension**: The dimension of the space in which the sensor operates (e.g., `2`, for 2D).
- **sensor_mounting_location**: The position `[x, y]` where the sensor is mounted on the robot (e.g., `[0, 0]`).
- **sensor_mounting_angle**: The angle `[θ]` at which the sensor is mounted, in radians (e.g., `[0]`).
- **has_noise**: Specifies whether the sensor data includes noise (e.g., `False`).

#### Usage Example 
Below is an example of how to instantiate and use the `LiDAR` class:
<details>
<summary>Click to expand</summary>
  

```python
# Define LiDAR parameters
lidar_params = AD(
    max_range=5,
    ray_num=100,
    scan_angle=[-torch.pi, torch.pi],
    update_rate=0.2,
    ray_sampling_rate=1000,
    space_dimension=2,
    sensor_mounting_location=[0, 0],
    sensor_mounting_angle=[0],
    has_noise=False
)

# Initialize the LiDAR class with a map, parameters, and dynamics model
lidar = Lidar(map, lidar_params, dynamics)

# Simulate for a specified time and collect sensor data
for _ in range(sim_time):
    point_cloud = lidar.update(x)

```
</details>




### LiDARMap Class




### ACC Presentation
<video src="https://github.com/user-attachments/assets/94da578b-bd7d-4ed6-b84b-bf8b1feb4feb" controls="controls" muted="muted" style="max-width:100%;"></video>


If you use this code or find it helpful in your research, please consider citing our paper:

```bibtex
@article{safari2024time,
  title={Time-Varying Soft-Maximum Barrier Functions for Safety in Unmapped and Dynamic Environments},
  author={Safari, Amirsaeid and Hoagg, Jesse B},
  journal={arXiv preprint arXiv:2409.01458},
  year={2024}
}
```

```bibtex
@INPROCEEDINGS{10644256,
  author={Safari, Amirsaeid and Hoagg, Jesse B.},
  booktitle={2024 American Control Conference (ACC)}, 
  title={Time-Varying Soft-Maximum Control Barrier Functions for Safety in an A Priori Unknown Environment}, 
  year={2024},
  volume={},
  number={},
  pages={3698-3703},
  keywords={Costs;Real-time systems;Safety;Robots;Optimization},
  doi={10.23919/ACC60939.2024.10644256}
}```


