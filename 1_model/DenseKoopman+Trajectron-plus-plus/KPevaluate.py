import sys
import os
import dill
import json
import argparse
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
import evaluation

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str, default='./models/VSC_kpspace_vel')  ##
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int, default=2000)
parser.add_argument("--data", help="full path to data file", type=str, default='../processed/VSC_KPTEST') ######
parser.add_argument("--output_path", help="KP_MOT20 KP_MOT20 KP_MOT20", type=str, default='./results/KP_VSC') ##
parser.add_argument("--node_type", help="node type to evaluate", type=str, default='PEDESTRIAN')
args = parser.parse_args()


def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')

    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams


if __name__ == "__main__":
    path_dir = args.data
    dirs = os.listdir(path_dir)
    # print(dirs)
    all_ade = []
    all_fde = []
    for file0 in dirs:
        file_path0 = os.path.join(path_dir, file0)

        with open(file_path0, 'rb') as f:
            env = dill.load(f, encoding='latin1')

        eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)

        if 'override_attention_radius' in hyperparams:
            for attention_radius_override in hyperparams['override_attention_radius']:
                node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
                env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

        scenes = env.scenes

        # print("-- Preparing Node Graph")
        for scene in tqdm(scenes):
            scene.calculate_scene_graph(env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])

        ph = hyperparams['prediction_horizon']
        max_hl = hyperparams['maximum_history_length']

        save_log_path = args.output_path
        save_name = ((file_path0).split('\\')[-1]).split('.')[0]
        # print(save_log_path, save_name)
        save_path = (os.path.join(save_log_path, save_name)) + '.txt'
        # print(save_path)
        with torch.no_grad():

            ############### BEST OF 20 ###############
            eval_ade_batch_errors = np.array([])
            eval_fde_batch_errors = np.array([])
            eval_kde_nll = np.array([])
            # print("-- Evaluating best of 20")
            predictions_all = {}
            for i, scene in enumerate(scenes):

                # print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
                timesteps = np.arange(0, 20)
                predictions = eval_stg.predict(scene,
                                               timesteps,
                                               ph,
                                               num_samples=20,
                                               min_history_timesteps=7,
                                               min_future_timesteps=12,
                                               z_mode=False,
                                               gmm_mode=False,
                                               full_dist=False)

                if not predictions:
                    continue

                if i == 0:
                    predictions_all = predictions
                elif i > 0:
                    for key1, value1 in predictions.items():
                        for key2, value2 in value1.items():
                            predictions_all[key1][key2] += predictions[key1][key2]

            batch_error_dict = evaluation.compute_batch_statistics(predictions_all,
                                                                   scene.dt,
                                                                   max_hl=max_hl,
                                                                   ph=ph,
                                                                   node_type_enum=env.NodeType,
                                                                   map=None,
                                                                   best_of=True,
                                                                   prune_ph_to_future=True)
            eval_ade_batch_errors = batch_error_dict[args.node_type]['ade']
            eval_fde_batch_errors = batch_error_dict[args.node_type]['fde']
            eval_kde_nll = batch_error_dict[args.node_type]['kde']
            ade = eval_ade_batch_errors
            fde = eval_fde_batch_errors
            with open(save_path, "w") as f:
                f.write(f"ADE:{ade}, FDE:{fde}")
                f.close()
            # print(f"{save_name}_ADE:", ade, "  FDE:", fde)
            all_fde.append(fde)
            all_ade.append(ade)
    ADE_ALL = np.mean(all_ade)
    FDE_ALL = np.mean(all_fde)
    print('Alldata--', f"ADE:{ADE_ALL}, FDE:{FDE_ALL}")

