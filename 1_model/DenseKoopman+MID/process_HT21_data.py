import pdb
import sys
import os
import numpy as np
import pandas as pd
import dill
import pickle

from environment import Environment, Scene, Node, derivative_of

desired_max_time = 100
pred_indices = [2, 3]
state_dim = 6
frame_diff = 10
desired_frame_diff = 1
# 0.4
dt = 0.4

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    }
}

def maybe_makedirs(path_to_create):
    """This function will create a directory, unless it exists already,
    at which point the function will return.
    The exception handling is necessary as it prevents a race condition
    from occurring.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise


def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name)

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        x = node.data.position.x.copy()
        y = node.data.position.y.copy()

        x, y = rotate_pc(np.array([x, y]), alpha)

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        data_dict = {('position', 'x'): x,
                     ('position', 'y'): y,
                     ('velocity', 'x'): vx,
                     ('velocity', 'y'): vy,
                     ('acceleration', 'x'): ax,
                     ('acceleration', 'y'): ay}

        node_data = pd.DataFrame(data_dict, columns=data_columns)

        node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep)

        scene_aug.nodes.append(node)
    return scene_aug




def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug


nl = 0
l = 0

data_folder_name = 'processed_data_HT21_All_kpspace'

maybe_makedirs(data_folder_name)
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

for desired_source in ['HT21_all_data_kpspace']:
    for data_class in ['test_0119', 'test_0120', 'test_0219', 'test_0220', 'test_0221', 'test_0222',
                       'test_0223', 'test_0224', 'test_0225', 'test_0226', 'test_0227', 'test_0228',
                       'test_0229', 'test_0230', 'test_0231', 'test_0232', 'test_0233', 'test_0234',
                       'test_0235', 'test_0236', 'test_0237', 'test_0238', 'test_0239', 'test_0240',
                       'test_0241', 'test_0242', 'test_0243', 'test_0244', 'test_0245', 'test_0319',
                       'test_0320', 'test_0321', 'test_0322', 'test_0323', 'test_0324', 'test_0325',
                       'test_0326', 'test_0327', 'test_0328', 'test_0329', 'test_0330', 'test_0331',
                       'test_0332', 'test_0333', 'test_0334', 'test_0335', 'test_0336', 'test_0337',
                       'test_0338', 'test_0339', 'test_0340', 'test_0341', 'test_0342', 'test_0343',
                       'test_0344', 'test_0345', 'test_0346', 'test_0347', 'test_0348', 'test_0349',
                       'test_0419', 'test_0420', 'test_0421', 'test_0422', 'test_0423', 'test_0424',
                       'test_0425', 'test_0426', 'test_0427', 'test_0428', 'test_0429', 'test_0430',
                       'test_0431', 'test_0432', 'test_0433', 'test_0434', 'test_0435', 'test_0436',
                       'test_0437', 'test_0438']:  #'test', 'train', 'val'
        env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
        env.attention_radius = attention_radius

        scenes = []
        data_dict_path = os.path.join(data_folder_name, '_'.join([desired_source, data_class]) + '.pkl')

        for subdir, dirs, files in os.walk(os.path.join('raw_data', desired_source, data_class)):
            for file in files:
                if file.endswith('.txt'):
                    input_data_dict = dict()
                    full_data_path = os.path.join(subdir, file)
                    print('At', full_data_path)

                    data = pd.read_csv(full_data_path, sep='\t', index_col=False, header=None)
                    data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
                    data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
                    data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')
                    data['frame_id'] = data['frame_id']
                    data['frame_id'] -= data['frame_id'].min()
                    data['node_type'] = 'PEDESTRIAN'
                    data['node_id'] = data['track_id'].astype(str)

                    data.sort_values('frame_id', inplace=True)

                    if data_class == "train":
                        data['pos_x'] = data['pos_x'] + 0.001 * np.random.normal(0,1)
                        data['pos_y'] = data['pos_y'] + 0.001 * np.random.normal(0,1)

                    data['pos_x'] = data['pos_x'] - data['pos_x'].mean()
                    data['pos_y'] = data['pos_y'] - data['pos_y'].mean()

                    max_timesteps = data['frame_id'].max()

                    scene = Scene(timesteps=max_timesteps+1, dt=dt, name=desired_source + "_" + data_class, aug_func=augment if data_class == 'train' else None)

                    for node_id in pd.unique(data['node_id']):

                        node_df = data[data['node_id'] == node_id]

                        node_values = node_df[['pos_x', 'pos_y']].values

                        if node_values.shape[0] < 2:
                            continue

                        new_first_idx = node_df['frame_id'].iloc[0]

                        x = node_values[:, 0]
                        y = node_values[:, 1]
                        vx = derivative_of(x, scene.dt)
                        vy = derivative_of(y, scene.dt)
                        ax = derivative_of(vx, scene.dt)
                        ay = derivative_of(vy, scene.dt)

                        data_dict = {('position', 'x'): x,
                                     ('position', 'y'): y,
                                     ('velocity', 'x'): vx,
                                     ('velocity', 'y'): vy,
                                     ('acceleration', 'x'): ax,
                                     ('acceleration', 'y'): ay}

                        node_data = pd.DataFrame(data_dict, columns=data_columns)
                        node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data)
                        node.first_timestep = new_first_idx
                        scene.nodes.append(node)

                    if data_class == 'train':
                        scene.augmented = list()
                        angles = np.arange(0, 360, 15)
                        for angle in angles:
                            scene.augmented.append(augment_scene(scene, angle))
                    print(scene)
                    scenes.append(scene)
        print(f'Processed {len(scenes):.2f} scene for data class {data_class}')

        env.scenes = scenes

        if len(scenes) > 0:
            with open(data_dict_path, 'wb') as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
exit()

