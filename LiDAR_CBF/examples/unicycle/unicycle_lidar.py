import torch
from attrdict import AttrDict as AD
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import os

from hocbf_composition.utils.make_map import Map
from hocbf_composition.safe_controls.closed_form_safe_control import MinIntervCFSafeControl

from LiDAR_CBF.examples.unicycle.unicycle_dynamics import UnicycleDynamics
from LiDAR_CBF.examples.unicycle.map_config import map_config
from LiDAR_CBF.examples.unicycle.unicycle_desired_control import desired_control
from LiDAR_CBF.lidar import Lidar
from LiDAR_CBF.make_online_map import *
from time import time
import datetime

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'Times'

torch.set_default_dtype(torch.float64)

# Control gains
control_gains = dict(k1=0.2, k2=1.0, k3=2.0)

# Barrier configs
cfg = AD(softmax_rho=30.0,
         softmin_rho=30.0,
         pos_barrier_rel_deg=2,
         vel_barrier_rel_deg=1,
         obstacle_alpha=[30.0],
         boundary_alpha=[1.0],
         velocity_alpha=[10],
         memory=5,
         cbf_synthesis='EllipseCBF',
         local_normalize=False,
         safe_dist=0.15,
         ellipse_width=0.8,
         smooth_function='SmoothStep',
         nu=2
         )

lidar_cfg = AD(max_range=5,
               ray_num=100,
               scan_angle=[-torch.pi, torch.pi],
               update_rate=0.2,
               ray_sampling_rate=1000,
               space_dimension=2,
               sensor_mounting_location=[0, 0],
               sensor_mounting_angle=[0],
               has_noise=False)


timestep = 0.01
sim_time = 10

# Goal positions
# goal_pos = torch.tensor(
#     [[3.0, 4.5]])

goal_pos = torch.tensor(
    [[3.0, 4.5], [-5.0, 0.0]])

# Index of the Robot for Plot Generation
robot_indices = [0]

make_animation = False



# Initial Conditions
# x0 = torch.tensor([[0, -1.0, -8.5, 0.0, torch.pi / 2]]).repeat(goal_pos.shape[0], 1)
x0 = torch.tensor([[0, -1.0, -8.5, 0.0, torch.pi / 2], [0, 0.0, 8.0, 0.0, -torch.pi / 2]], dtype=torch.float64)
# x0 = torch.tensor([[0, -1.0, -8.5, 0.0, pi / 2], [0, -1.0, -8.5, 0.0, pi / 2]])




# Instantiate dynamics
dynamics = UnicycleDynamics(state_dim=4, action_dim=2)

# Make barrier from map_
map_ = Map(barriers_info=map_config, dynamics=dynamics, cfg=cfg)

lidar = Lidar(map_, lidar_cfg, dynamics)
lidarmap = LidarMap(lidar, dynamics, cfg)

# Make safety filter and assign dynamics
safety_filter = MinIntervCFSafeControl(
    action_dim=lidarmap.dynamics.action_dim,
    alpha=lambda x: 50 * x,
    params=AD(slack_gain=200,
              use_softplus=True,
              softplus_gain=2.0)
).assign_dynamics(dynamics=lidarmap.dynamics)


# assign desired control based on the goal positions
safety_filter.assign_desired_control(
    desired_control=lambda x: vectorize_tensors(partial(desired_control, goal_pos=goal_pos, **control_gains)(x[..., 1:])))



all_trajs = []
trajs = x0.unsqueeze(0)
start_time = time()
for i in range(int(sim_time / lidar_cfg.update_rate) + 1):
    point_cloud = lidar.update(trajs[-1, :, 1:])
    lidarmap.update(trajs[-1, :, 1:3].unsqueeze(0), point_cloud)
    safety_filter.assign_state_barrier(lidarmap.barrier)
    trajs = safety_filter.get_safe_optimal_trajs_zoh(x0=trajs[[-1, ...]], sim_time=lidar_cfg.update_rate,
                                                     timestep=timestep, method='euler')
    all_trajs.append(trajs[0:-1, ...])

    if torch.norm(trajs[-1, :, 0:2] - goal_pos) <= 1e-2:
        break

print(time() - start_time)

# Rearrange trajs
all_trajs = torch.cat(all_trajs, dim=0)
all_trajs = [torch.vstack(t.split(lidarmap.dynamics.state_dim)) for t in torch.hstack([tt for tt in all_trajs])]

pw_barrier = lidarmap.make_piecewise_barrier(robot_indices=robot_indices)




# Get actions and h values along the trajs
actions = []
constraint_val = []
psi0 = []
psi1 = []
for i, idx in enumerate(robot_indices):
    safety_filter.assign_state_barrier(pw_barrier[i])
    des_ctrl = lambda x: vectorize_tensors(
        partial(desired_control, goal_pos=goal_pos[i].repeat(x.shape[0], 1), **control_gains)(x[..., 1:]))
    safety_filter.assign_desired_control(
        desired_control=des_ctrl
    )
    action, info = safety_filter.safe_optimal_control(all_trajs[idx], ret_info=True)
    constraint_val.append(info['constraint_at_u'])
    actions.append(action)
    psi0.append(safety_filter.barrier.barrier(all_trajs[idx]))
    psi1.append(safety_filter.barrier.hocbf(all_trajs[idx]))


############
#  Plots   #
############

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
current_path = os.getcwd()
plot_folder = 'unicycle_lidar_plot'
saving_path = os.path.join(current_path, plot_folder)

# Countor plot
x = torch.linspace(-14, 14, 500)
y = torch.linspace(-14, 14, 500)
X, Y = torch.meshgrid(x, y)
points = torch.column_stack((X.flatten(), Y.flatten()))
points = torch.column_stack((points, torch.zeros(points.shape)))
Z = map_.map_barrier.min_barrier(points).reshape(X.shape)
plt.figure(figsize=(8, 6), dpi=100)
plt.contour(X, Y, Z, levels=[0], colors='k')
# plt.contour(X, Y, Z1, levels=[0], colors='b')
plt.xlabel(r'$q_x$')
plt.ylabel(r'$q_y$')
for i in range(len(all_trajs)):
    plt.plot(goal_pos[i][0], goal_pos[i][1], '*', markersize=10, color='red', label='Goal' if i == 0 else None)

    plt.plot(all_trajs[i][:, 1], all_trajs[i][:, 2], label='Trajectories' if i == 0 else None, color='blue')

    plt.plot(all_trajs[i][0, 1], all_trajs[i][0, 2], 'o', markersize=8, color='blue', label='Initial State' if i==0 else None)

    plt.legend()
# plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(saving_path + f'/contour_plot_{current_time}.png')
plt.show()




for i, idx in enumerate(robot_indices):
    # Calculate time array based on the number of data points and timestep
    time = all_trajs[idx][..., 0]


    # Create subplot for trajs and action variables
    fig, axs = plt.subplots(8, 1, figsize=(8, 10))

    # Plot trajs variables
    axs[0].plot(time, all_trajs[idx][:, 1].detach(), label=r'$q_x$', color='red')
    axs[0].plot(time, all_trajs[idx][:, 2].detach(), label=r'$q_y$', color='blue')
    axs[0].set_ylabel(r'$q_x, q_y$', fontsize=14)
    axs[0].legend(fontsize=14)
    axs[0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

    axs[1].plot(time, all_trajs[idx][:, 3].detach(), label='v', color='black')
    axs[1].set_ylabel(r'$v$', fontsize=14)

    axs[2].plot(time, all_trajs[i][:, 4].detach(), label='theta', color='black')
    axs[2].set_ylabel(r'$\theta$', fontsize=14)

    # Plot actions
    axs[3].plot(time, actions[i][:, 0].detach(), label='u_1', color='black')
    axs[3].set_ylabel(r'$u_1$', fontsize=14)

    axs[4].plot(time, actions[i][:, 1].detach(), label='u_2', color='black')
    axs[4].set_ylabel(r'$u_2$', fontsize=14)

    # Plot online h
    axs[5].plot(time, psi0[i].detach(), label='barrier', color='black')
    axs[5].set_ylabel(r'$\psi_0$', fontsize=14)

    # Plot barrier values
    axs[6].plot(time, psi1[i].detach(), label='barrier', color='black')
    axs[6].set_ylabel(r'$\psi_1$', fontsize=14)


    # Plot constraint values
    axs[7].plot(time, (constraint_val[i]), label='barrier', color='black')
    axs[7].set_ylabel(r'$\dot{\psi_1} + \alpha \psi_1$', fontsize=14)



    axs[7].set_xlabel(r'$t~(\rm {s})$', fontsize=14)


    for i in range(5):
        axs[i].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

    # Multiply x-axis labels by timestep value (0.001)
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=12)

    # Adjust layout and save the combined plot
    plt.tight_layout()
    plt.savefig(saving_path + f'/combined_plot_{idx}_{current_time}.png')

    # Show the plots
    plt.show()


# Countor Plot Gif
if make_animation:
    cmap = plt.cm.Greens

    # Pre-compute grid points
    x = torch.linspace(-14, 14, 100)
    y = torch.linspace(-14, 14, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack((X.flatten(), Y.flatten(), torch.zeros_like(X.flatten())), dim=1)

    # Pre-compute Z for the map barrier
    Z = map_.map_barrier.min_barrier(points).reshape(X.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    metadata = dict(title='Movie', artist='codinglikemad')
    writer = PillowWriter(fps=240, metadata=metadata)

    with writer.saving(fig, f'{saving_path}/contour_plot_animation_{current_time}.gif', 100):
        times = all_trajs[0][..., 0]
        for traj_idx, time in enumerate(times):
            ax.clear()
            ax.set_xlabel(r'$q_x$')
            ax.set_ylabel(r'$q_y$')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])

            # Plot map barrier contour
            ax.contour(X, Y, Z, levels=[0], colors='k')

            for i, idx in enumerate(robot_indices):
                time_aug_points = torch.cat([time.expand(points.shape[0], 1), points[:, :2], torch.zeros_like(points)],
                                            dim=1).unsqueeze(0)
                ZZ = lidarmap.pw_barrier[i].barrier(time_aug_points)[0, :].reshape(X.shape)

                # Compute contour levels based on the current ZZ
                contour_levels = torch.linspace(0.0, torch.max(ZZ), 100)

                # Plot contours
                ax.contour(X, Y, ZZ, levels=[0], colors='g')
                ax.contourf(X, Y, ZZ, levels=contour_levels, cmap=cmap)

                # Plot start, goal, and trajectory
                ax.plot(all_trajs[idx][0, 1], all_trajs[idx][0, 2], 'o', markersize=8)
                ax.plot(goal_pos[i][0], goal_pos[i][1], '*', markersize=10, color='red')
                ax.plot(all_trajs[i][:traj_idx + 1, 1], all_trajs[i][:traj_idx + 1, 2], color='blue')

            writer.grab_frame()
            print(f'{traj_idx}')




