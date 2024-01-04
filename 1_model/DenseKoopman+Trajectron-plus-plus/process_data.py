import sys
import os
import numpy as np
import pandas as pd
import dill

# sys.path.append(".../trajectron")
from environment import Environment, Scene, Node
from utils import maybe_makedirs
from environment import derivative_of

desired_max_time = 100
pred_indices = [2, 3]
state_dim = 6
frame_diff = 10
desired_frame_diff = 1
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
maybe_makedirs('../processed')
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
for desired_source in ['VSC_all_data_kpspace/kpspace_test']: ################################################################################
    for data_class in ['test_07219', 'test_07220', 'test_07319', 'test_07320', 'test_07419', 'test_07420',
                       'test_07619', 'test_07620', 'test_07621', 'test_07622', 'test_07623',
                       'test_07624', 'test_07625', 'test_07626', 'test_07627', 'test_07628', 'test_07629',
                       'test_07630', 'test_07719', 'test_10019', 'test_10020', 'test_10021', 'test_10022',
                       'test_10023', 'test_10024', 'test_10025', 'test_10026', 'test_10027', 'test_10028',
                       'test_10029', 'test_11119', 'test_11120', 'test_11121', 'test_11122', 'test_17119',
                       'test_17120', 'test_17121', 'test_17122', 'test_17123', 'test_17124', 'test_17125',
                       'test_28619', 'test_28620', 'test_28621', 'test_34819', 'test_34820', 'test_34821',
                       'test_34822', 'test_34823', 'test_34824', 'test_34825', 'test_34826', 'test_46419',
                       'test_46420', 'test_46421', 'test_46422', 'test_46423', 'test_49219', 'test_49220',
                       'test_49221', 'test_49222', 'test_49223', 'test_49224', 'test_49225', 'test_49226',
                       'test_49227', 'test_49228', 'test_49229', 'test_49230', 'test_49231', 'test_49232',
                       'test_49233', 'test_49234', 'test_49235', 'test_49236', 'test_49237', 'test_49238',
                       'test_49239', 'test_49240', 'test_49241', 'test_49242', 'test_49243', 'test_49244',
                       'test_49245', 'test_49246', 'test_49319', 'test_49320', 'test_49321', 'test_49322',
                       'test_49324', 'test_49325', 'test_49326', 'test_49327', 'test_49328',
                       'test_49329', 'test_49330', 'test_49331', 'test_49332', 'test_49333', 'test_49334',
                       'test_49335', 'test_49336', 'test_49337', 'test_49338', 'test_49339', 'test_49340',
                       'test_49341', 'test_49342', 'test_49343', 'test_49344', 'test_49345', 'test_49346',
                       'test_49347', 'test_49348', 'test_49349', 'test_49350', 'test_49351', 'test_49352',
                       'test_49353', 'test_49419', 'test_49420', 'test_49421', 'test_49422', 'test_49423',
                       'test_49424', 'test_49425', 'test_49426', 'test_49427', 'test_49428', 'test_49429',
                       'test_49430', 'test_49431', 'test_49432', 'test_49433', 'test_49434', 'test_49435',
                       'test_49436', 'test_49437', 'test_49438', 'test_49439', 'test_49440', 'test_49441',
                       'test_49442', 'test_49443', 'test_49444', 'test_49445', 'test_49446', 'test_49447',
                       'test_49448']:  #'train'
        env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
        env.attention_radius = attention_radius

        scenes = []
        # data_dict_path = os.path.join('../processed/MOT20_KPTEST', '_'.join([desired_source, data_class]) + '.pkl')#################################
        data_dict_path = os.path.join('../processed/VSC_KPTEST', data_class + '.pkl')
        for subdir, dirs, files in os.walk(os.path.join('raw', desired_source, data_class)):
            for file in files:
                if file.endswith('.txt'):
                    input_data_dict = dict()
                    full_data_path = os.path.join(subdir, file)
                    print('At', full_data_path)

                    data = pd.read_csv(full_data_path, sep='\t', index_col=False, header=None)
                    data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
                    data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
                    data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')

                    data['frame_id'] = data['frame_id'] // 10

                    data['frame_id'] -= data['frame_id'].min()

                    data['node_type'] = 'PEDESTRIAN'
                    data['node_id'] = data['track_id'].astype(str)
                    data.sort_values('frame_id', inplace=True)

                    # Mean Position
                    data['pos_x'] = data['pos_x'] - data['pos_x'].mean()
                    data['pos_y'] = data['pos_y'] - data['pos_y'].mean()

                    max_timesteps = data['frame_id'].max()

                    scene = Scene(timesteps=max_timesteps+1, dt=dt, name=desired_source + "_" + data_class, aug_func=augment if data_class == 'train' else None)

                    for node_id in pd.unique(data['node_id']):

                        node_df = data[data['node_id'] == node_id]
                        assert np.all(np.diff(node_df['frame_id']) == 1)

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
                        angles = np.arange(0, 360, 15) if data_class == 'train' else [0]
                        for angle in angles:
                            scene.augmented.append(augment_scene(scene, angle))

                    print(scene)
                    scenes.append(scene)
        print(f'Processed {len(scenes):.2f} scene for data class {data_class}')

        env.scenes = scenes

        if len(scenes) > 0:
            with open(data_dict_path, 'wb') as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)

print(f"Linear: {l}")
print(f"Non-Linear: {nl}")