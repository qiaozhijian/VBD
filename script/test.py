import os
import glob
import pickle
import torch
import csv
import mediapy
import argparse
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

# set tf to cpu only
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import jax
jax.config.update('jax_platform_name', 'cpu')

# utils
from vbd.data.dataset import WaymaxTestDataset
from vbd.model.utils import set_seed
from vbd.sim_agent.sim_actor import VBDTest, sample_to_action
from vbd.waymax_visualization.plotting import plot_state

# waymax
from waymax import dynamics
from waymax import datatypes
from waymax import env as _env
from waymax import visualization
from waymax.config import EnvironmentConfig, ObjectType
from waymax.config import DatasetConfig, DataFormat
from waymax.metrics.comfort import KinematicsInfeasibilityMetric
from waymax.metrics import OverlapMetric, LogDivergenceMetric
from vbd.sim_agent.waymax_metrics import OffroadMetric, WrongWayMetric
from vbd.sim_agent.waymax_env import WaymaxEnvironment
from vbd.data.waymax_utils import create_iter


## Parameters
CURRENT_TIME_INDEX = 10
N_SIM_AGENTS = 32
N_SIMULATION_STEPS = 80


## Set up Waynax Environment
env_config = EnvironmentConfig(
    # Ensure that the sim agent can control all valid objects.
    controlled_object=ObjectType.VALID,
    max_num_objects=N_SIM_AGENTS,
    allow_new_objects_after_warmup=False
)

dynamics_model = dynamics.StateDynamics()

env = WaymaxEnvironment(
    dynamics_model=dynamics_model,
    config=env_config,
)

dataset = WaymaxTestDataset(
    data_dir=None,
    anchor_path='./vbd/data/cluster_64_center_dict.pkl',
    max_object=N_SIM_AGENTS
)


## Calculate metrics
def calculate_metrics(metrics, modeled_indices):
    offroad = []
    collision = []
    log_divergence = []
    wrong_way = []
    kinematic_infeasibility = []

    for i in modeled_indices:
        is_offroad = metrics[0]['offroad'].value[i]
        is_collision = metrics[0]['overlap'].value[i]
        col, off, kin, div, wrw = [], [], [], [], []

        for t in range(len(metrics)):
            valid = metrics[t]['log_divergence'].valid[i]
            div.append((metrics[t]['log_divergence'].value[i] * valid).item())
            col.append((metrics[t]['overlap'].value[i]).item())
            off.append((metrics[t]['offroad'].value[i]).item())
            kin.append((metrics[t]['kinematic_infeasibility'].value[i]).item())
            wrw.append((metrics[t]['wrong_way'].value[i]).item())

        collision.append(np.any(col) if not is_collision else False)
        offroad.append(np.any(off) if not is_offroad else False)
        wrong_way.append(np.sum(wrw) > 10)
        log_divergence.append(np.mean(div))
        kinematic_infeasibility.append(np.mean(kin))

    metrics = {
        'collision': np.mean(collision),
        'offroad': np.mean(offroad),
        'wrong_way': np.mean(wrong_way),
        'log_divergence': np.mean(log_divergence),
        'kinematic_infeasibility': np.mean(kinematic_infeasibility),
    }

    return metrics


## Begin Simulation
def run_simulation(args):
    ## Load model
    vbd = VBDTest.load_from_checkpoint(args.model_path, args.device)
    vbd.reset_agent_length(N_SIM_AGENTS)
    vbd.eval()
    set_seed(args.seed)  
    
    # Load testing scenarios
    config = DatasetConfig(
        path=args.test_path,
        max_num_objects=N_SIM_AGENTS,
        repeat=1,
        max_num_rg_points=30000,
        data_format=DataFormat.TFRECORD
    )

    data_iter = create_iter(config)

    # Save results
    SAVE_PATH = f'testing_results/test_{args.test_mode}/{args.seed}'
    os.makedirs(SAVE_PATH, exist_ok=True)

    with open(os.path.join(SAVE_PATH, 'metrics.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['scenario_id', 'collision', 'offroad', 'wrong_way', 
                        'log_divergence', 'kinematic_infeasibility'])

    # Begin simulation
    for scenario_id, scenario in data_iter:  
        print(f"Running scenario {scenario_id}...")
        initial_state = current_state = env.reset(scenario)
        log_states = [initial_state]
        log_metrics = []    
        is_valid = scenario.object_metadata.is_valid
        is_controlled = is_valid[:N_SIM_AGENTS]

        # Run the simulated scenarios.
        for t in (range(initial_state.remaining_timesteps)):
            i = t % args.replan

            if i == 0:
                print("Replan at ", current_state.timestep)

                with torch.no_grad():
                    sample = dataset.process_scenario(current_state, current_state.timestep, use_log=False)
                    batch = dataset.__collate_fn__([sample])

                    if args.test_mode == 'diffusion':
                        pred = vbd.sample_denoiser(batch)
                        pred_traj = pred['denoised_trajs'].cpu().numpy()[0]

                    elif args.test_mode == 'prior':
                        pred = vbd.inference_predictor(batch)
                        scores = pred['goal_scores'][0].softmax(dim=-1)
                        trajs = pred['goal_trajs'][0]
                        sampled_idx = torch.multinomial(scores, 1).squeeze()
                        pred_traj = trajs[torch.arange(sampled_idx.shape[0]), sampled_idx].cpu().numpy()

                    else:
                        raise NotImplementedError

            sample = pred_traj[:, i, :]
            action = sample_to_action(sample, is_controlled, None, N_SIM_AGENTS)
            current_state = env.step_sim_agent(current_state, [action])
            log_states.append(current_state)

            # Run metrics
            overlap = OverlapMetric().compute(current_state)
            offroad = OffroadMetric().compute(current_state)
            wrongway = WrongWayMetric().compute(current_state)
            log_divergence = LogDivergenceMetric().compute(current_state)
            kinematic_infeasibility = KinematicsInfeasibilityMetric().compute(current_state)
            log_metrics.append({
                'overlap': overlap,
                'offroad': offroad,
                'wrong_way': wrongway,
                'log_divergence': log_divergence,
                'kinematic_infeasibility': kinematic_infeasibility
            })
        

        # Calculate metrics
        modeled_indices = jax.numpy.where(is_controlled)[0].tolist()
        metrics = calculate_metrics(log_metrics, modeled_indices)
        with open(os.path.join(SAVE_PATH, 'metrics.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([scenario_id, metrics['collision'], metrics['offroad'], metrics['wrong_way'], 
                            metrics['log_divergence'], metrics['kinematic_infeasibility']])

        # Visualize results
        if args.save_simulation:
            sim_images = []
            for state in log_states:
                img = plot_state(state)
                sim_images.append(img)

            with open(os.path.join(SAVE_PATH, f'{scenario_id}_sim.pkl'), 'wb') as f:
                pickle.dump(state.sim_trajectory, f)

            with mediapy.set_show_save_dir(SAVE_PATH):
                mediapy.show_video(sim_images, title=f'{scenario_id}', fps=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--replan', type=int, default=10, help='Replan frequency')
    parser.add_argument('--test_mode', type=str, default='diffusion')
    parser.add_argument('--save_simulation', action='store_true')

    args = parser.parse_args()
    run_simulation(args)