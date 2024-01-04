# import warnings
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from ..model.components import *
# from ..model.model_utils import *
# from ..model.dynamics import *
# from ..environment.scene_graph import DirectedEdge
#
#
# class MultimodalGenerativeCVAE(object):
#     def __init__(self,
#                  env,
#                  node_type,
#                  model_registrar,
#                  hyperparams,
#                  device,
#                  edge_types,
#                  log_writer=None):
#         self.hyperparams = hyperparams
#         self.env = env
#         self.node_type = node_type
#         self.model_registrar = model_registrar
#         self.log_writer = log_writer
#         self.device = device
#         self.edge_types = [edge_type for edge_type in edge_types if edge_type[0] is node_type]
#         self.curr_iter = 0
#
#         self.node_modules = dict()
#
#         self.min_hl = self.hyperparams['minimum_history_length']
#         self.max_hl = self.hyperparams['maximum_history_length']
#         self.ph = self.hyperparams['prediction_horizon']
#         self.state = self.hyperparams['state']
#         self.pred_state = self.hyperparams['pred_state'][node_type]
#         self.state_length = int(np.sum([len(entity_dims) for entity_dims in self.state[node_type].values()]))
#         if self.hyperparams['incl_robot_node']:
#             self.robot_state_length = int(
#                 np.sum([len(entity_dims) for entity_dims in self.state[env.robot_type].values()])
#             )
#         self.pred_state_length = int(np.sum([len(entity_dims) for entity_dims in self.pred_state.values()]))
#
#         edge_types_str = [DirectedEdge.get_str_from_types(*edge_type) for edge_type in self.edge_types]
#         self.create_graphical_model(edge_types_str)
#
#         dynamic_class = getattr(Dynamic, hyperparams['dynamic'][self.node_type]['name'])
#         dyn_limits = hyperparams['dynamic'][self.node_type]['limits']
#         self.dynamic = dynamic_class(self.env.scenes[0].dt, dyn_limits, device,
#                                      self.model_registrar, self.x_size, self.node_type)
#
#     def set_curr_iter(self, curr_iter):
#         self.curr_iter = curr_iter
#
#     def add_submodule(self, name, model_if_absent):
#         self.node_modules[name] = self.model_registrar.get_model(name, model_if_absent)
#
#     def clear_submodules(self):
#         self.node_modules.clear()
#
#     def create_node_models(self):
#         ############################
#         #   Node History Encoder   #
#         ############################
#         self.add_submodule(self.node_type + '/node_history_encoder',
#                            model_if_absent=nn.LSTM(input_size=self.state_length,
#                                                    hidden_size=self.hyperparams['enc_rnn_dim_history'],
#                                                    batch_first=True))
#
#         ###########################
#         #   Node Future Encoder   #
#         ###########################
#         # We'll create this here, but then later check if in training mode.
#         # Based on that, we'll factor this into the computation graph (or not).
#         self.add_submodule(self.node_type + '/node_future_encoder',
#                            model_if_absent=nn.LSTM(input_size=self.pred_state_length,
#                                                    hidden_size=self.hyperparams['enc_rnn_dim_future'],
#                                                    bidirectional=True,
#                                                    batch_first=True))
#         # These are related to how you initialize states for the node future encoder.
#         self.add_submodule(self.node_type + '/node_future_encoder/initial_h',
#                            model_if_absent=nn.Linear(self.state_length,
#                                                      self.hyperparams['enc_rnn_dim_future']))
#         self.add_submodule(self.node_type + '/node_future_encoder/initial_c',
#                            model_if_absent=nn.Linear(self.state_length,
#                                                      self.hyperparams['enc_rnn_dim_future']))
#
#         ############################
#         #   Robot Future Encoder   #
#         ############################
#         # We'll create this here, but then later check if we're next to the robot.
#         # Based on that, we'll factor this into the computation graph (or not).
#         if self.hyperparams['incl_robot_node']:
#             self.add_submodule('robot_future_encoder',
#                                model_if_absent=nn.LSTM(input_size=self.robot_state_length,
#                                                        hidden_size=self.hyperparams['enc_rnn_dim_future'],
#                                                        bidirectional=True,
#                                                        batch_first=True))
#             # These are related to how you initialize states for the robot future encoder.
#             self.add_submodule('robot_future_encoder/initial_h',
#                                model_if_absent=nn.Linear(self.robot_state_length,
#                                                          self.hyperparams['enc_rnn_dim_future']))
#             self.add_submodule('robot_future_encoder/initial_c',
#                                model_if_absent=nn.Linear(self.robot_state_length,
#                                                          self.hyperparams['enc_rnn_dim_future']))
#
#         if self.hyperparams['edge_encoding']:
#             ##############################
#             #   Edge Influence Encoder   #
#             ##############################
#             # NOTE: The edge influence encoding happens during calls
#             # to forward or incremental_forward, so we don't create
#             # a model for it here for the max and sum variants.
#             if self.hyperparams['edge_influence_combine_method'] == 'bi-rnn':
#                 self.add_submodule(self.node_type + '/edge_influence_encoder',
#                                    model_if_absent=nn.LSTM(input_size=self.hyperparams['enc_rnn_dim_edge'],
#                                                            hidden_size=self.hyperparams['enc_rnn_dim_edge_influence'],
#                                                            bidirectional=True,
#                                                            batch_first=True))
#
#                 # Four times because we're trying to mimic a bi-directional
#                 # LSTM's output (which, here, is c and h from both ends).
#                 self.eie_output_dims = 4 * self.hyperparams['enc_rnn_dim_edge_influence']
#
#             elif self.hyperparams['edge_influence_combine_method'] == 'attention':
#                 # Chose additive attention because of https://arxiv.org/pdf/1703.03906.pdf
#                 # We calculate an attention context vector using the encoded edges as the "encoder"
#                 # (that we attend _over_)
#                 # and the node history encoder representation as the "decoder state" (that we attend _on_).
#                 self.add_submodule(self.node_type + '/edge_influence_encoder',
#                                    model_if_absent=AdditiveAttention(
#                                        encoder_hidden_state_dim=self.hyperparams['enc_rnn_dim_edge_influence'],
#                                        decoder_hidden_state_dim=self.hyperparams['enc_rnn_dim_history']))
#
#                 self.eie_output_dims = self.hyperparams['enc_rnn_dim_edge_influence']
#
#         ###################
#         #   Map Encoder   #
#         ###################
#         if self.hyperparams['use_map_encoding']:
#             if self.node_type in self.hyperparams['map_encoder']:
#                 me_params = self.hyperparams['map_encoder'][self.node_type]
#                 self.add_submodule(self.node_type + '/map_encoder',
#                                    model_if_absent=CNNMapEncoder(me_params['map_channels'],
#                                                                  me_params['hidden_channels'],
#                                                                  me_params['output_size'],
#                                                                  me_params['masks'],
#                                                                  me_params['strides'],
#                                                                  me_params['patch_size']))
#
#         ################################
#         #   Discrete Latent Variable   #
#         ################################
#         self.latent = DiscreteLatent(self.hyperparams, self.device)
#
#         ######################################################################
#         #   Various Fully-Connected Layers from Encoder to Latent Variable   #
#         ######################################################################
#         # Node History Encoder
#         x_size = self.hyperparams['enc_rnn_dim_history']
#         if self.hyperparams['edge_encoding']:
#             #              Edge Encoder
#             x_size += self.eie_output_dims
#         if self.hyperparams['incl_robot_node']:
#             #              Future Conditional Encoder
#             x_size += 4 * self.hyperparams['enc_rnn_dim_future']
#         if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
#             #              Map Encoder
#             x_size += self.hyperparams['map_encoder'][self.node_type]['output_size']
#
#         z_size = self.hyperparams['N'] * self.hyperparams['K']
#
#         if self.hyperparams['p_z_x_MLP_dims'] is not None:
#             self.add_submodule(self.node_type + '/p_z_x',
#                                model_if_absent=nn.Linear(x_size, self.hyperparams['p_z_x_MLP_dims']))
#             hx_size = self.hyperparams['p_z_x_MLP_dims']
#         else:
#             hx_size = x_size
#
#         self.add_submodule(self.node_type + '/hx_to_z',
#                            model_if_absent=nn.Linear(hx_size, self.latent.z_dim))
#
#         if self.hyperparams['q_z_xy_MLP_dims'] is not None:
#             self.add_submodule(self.node_type + '/q_z_xy',
#                                #                                           Node Future Encoder
#                                model_if_absent=nn.Linear(x_size + 4 * self.hyperparams['enc_rnn_dim_future'],
#                                                          self.hyperparams['q_z_xy_MLP_dims']))
#             hxy_size = self.hyperparams['q_z_xy_MLP_dims']
#         else:
#             #                           Node Future Encoder
#             hxy_size = x_size + 4 * self.hyperparams['enc_rnn_dim_future']
#
#         self.add_submodule(self.node_type + '/hxy_to_z',
#                            model_if_absent=nn.Linear(hxy_size, self.latent.z_dim))
#
#         ####################
#         #   Decoder LSTM   #
#         ####################
#         if self.hyperparams['incl_robot_node']:
#             decoder_input_dims = self.pred_state_length + self.robot_state_length + z_size + x_size
#         else:
#             decoder_input_dims = self.pred_state_length + z_size + x_size
#
#         self.add_submodule(self.node_type + '/decoder/state_action',
#                            model_if_absent=nn.Sequential(
#                                nn.Linear(self.state_length, self.pred_state_length)))
#
#         self.add_submodule(self.node_type + '/decoder/rnn_cell',
#                            model_if_absent=nn.GRUCell(decoder_input_dims, self.hyperparams['dec_rnn_dim']))
#         self.add_submodule(self.node_type + '/decoder/initial_h',
#                            model_if_absent=nn.Linear(z_size + x_size, self.hyperparams['dec_rnn_dim']))
#
#         ###################
#         #   Decoder GMM   #
#         ###################
#         self.add_submodule(self.node_type + '/decoder/proj_to_GMM_log_pis',
#                            model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
#                                                      self.hyperparams['GMM_components']))
#         self.add_submodule(self.node_type + '/decoder/proj_to_GMM_mus',
#                            model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
#                                                      self.hyperparams['GMM_components'] * self.pred_state_length))
#         self.add_submodule(self.node_type + '/decoder/proj_to_GMM_log_sigmas',
#                            model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
#                                                      self.hyperparams['GMM_components'] * self.pred_state_length))
#         self.add_submodule(self.node_type + '/decoder/proj_to_GMM_corrs',
#                            model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
#                                                      self.hyperparams['GMM_components']))
#
#         self.x_size = x_size
#         self.z_size = z_size
#
#     def create_edge_models(self, edge_types):
#         for edge_type in edge_types:
#             neighbor_state_length = int(
#                 np.sum([len(entity_dims) for entity_dims in self.state[edge_type.split('->')[1]].values()]))
#             if self.hyperparams['edge_state_combine_method'] == 'pointnet':
#                 self.add_submodule(edge_type + '/pointnet_encoder',
#                                    model_if_absent=nn.Sequential(
#                                        nn.Linear(self.state_length, 2 * self.state_length),
#                                        nn.ReLU(),
#                                        nn.Linear(2 * self.state_length, 2 * self.state_length),
#                                        nn.ReLU()))
#
#                 edge_encoder_input_size = 2 * self.state_length + self.state_length
#
#             elif self.hyperparams['edge_state_combine_method'] == 'attention':
#                 self.add_submodule(self.node_type + '/edge_attention_combine',
#                                    model_if_absent=TemporallyBatchedAdditiveAttention(
#                                        encoder_hidden_state_dim=self.state_length,
#                                        decoder_hidden_state_dim=self.state_length))
#                 edge_encoder_input_size = self.state_length + neighbor_state_length
#
#             else:
#                 edge_encoder_input_size = self.state_length + neighbor_state_length
#
#             self.add_submodule(edge_type + '/edge_encoder',
#                                model_if_absent=nn.LSTM(input_size=edge_encoder_input_size,
#                                                        hidden_size=self.hyperparams['enc_rnn_dim_edge'],
#                                                        batch_first=True))
#
#     def create_graphical_model(self, edge_types):
#         """
#         Creates or queries all trainable components.
#
#         :param edge_types: List containing strings for all possible edge types for the node type.
#         :return: None
#         """
#         self.clear_submodules()
#
#         ############################
#         #   Everything but Edges   #
#         ############################
#         self.create_node_models()
#
#         #####################
#         #   Edge Encoders   #
#         #####################
#         if self.hyperparams['edge_encoding']:
#             self.create_edge_models(edge_types)
#
#         for name, module in self.node_modules.items():
#             module.to(self.device)
#
#     def create_new_scheduler(self, name, annealer, annealer_kws, creation_condition=True):
#         value_scheduler = None
#         rsetattr(self, name + '_scheduler', value_scheduler)
#         if creation_condition:
#             annealer_kws['device'] = self.device
#             value_annealer = annealer(annealer_kws)
#             rsetattr(self, name + '_annealer', value_annealer)
#
#             # This is the value that we'll update on each call of
#             # step_annealers().
#             rsetattr(self, name, value_annealer(0).clone().detach())
#             dummy_optimizer = optim.Optimizer([rgetattr(self, name)], {'lr': value_annealer(0).clone().detach()})
#             rsetattr(self, name + '_optimizer', dummy_optimizer)
#
#             value_scheduler = CustomLR(dummy_optimizer,
#                                        value_annealer)
#             rsetattr(self, name + '_scheduler', value_scheduler)
#
#         self.schedulers.append(value_scheduler)
#         self.annealed_vars.append(name)
#
#     def set_annealing_params(self):
#         self.schedulers = list()
#         self.annealed_vars = list()
#
#         self.create_new_scheduler(name='kl_weight',
#                                   annealer=sigmoid_anneal,
#                                   annealer_kws={
#                                       'start': self.hyperparams['kl_weight_start'],
#                                       'finish': self.hyperparams['kl_weight'],
#                                       'center_step': self.hyperparams['kl_crossover'],
#                                       'steps_lo_to_hi': self.hyperparams['kl_crossover'] / self.hyperparams[
#                                           'kl_sigmoid_divisor']
#                                   })
#
#         self.create_new_scheduler(name='latent.temp',
#                                   annealer=exp_anneal,
#                                   annealer_kws={
#                                       'start': self.hyperparams['tau_init'],
#                                       'finish': self.hyperparams['tau_final'],
#                                       'rate': self.hyperparams['tau_decay_rate']
#                                   })
#
#         self.create_new_scheduler(name='latent.z_logit_clip',
#                                   annealer=sigmoid_anneal,
#                                   annealer_kws={
#                                       'start': self.hyperparams['z_logit_clip_start'],
#                                       'finish': self.hyperparams['z_logit_clip_final'],
#                                       'center_step': self.hyperparams['z_logit_clip_crossover'],
#                                       'steps_lo_to_hi': self.hyperparams['z_logit_clip_crossover'] / self.hyperparams[
#                                           'z_logit_clip_divisor']
#                                   },
#                                   creation_condition=self.hyperparams['use_z_logit_clipping'])
#
#     def step_annealers(self):
#         # This should manage all of the step-wise changed
#         # parameters automatically.
#         for idx, annealed_var in enumerate(self.annealed_vars):
#             if rgetattr(self, annealed_var + '_scheduler') is not None:
#                 # First we step the scheduler.
#                 with warnings.catch_warnings():  # We use a dummy optimizer: Warning because no .step() was called on it
#                     warnings.simplefilter("ignore")
#                     rgetattr(self, annealed_var + '_scheduler').step()
#
#                 # Then we set the annealed vars' value.
#                 rsetattr(self, annealed_var, rgetattr(self, annealed_var + '_optimizer').param_groups[0]['lr'])
#
#         self.summarize_annealers()
#
#     def summarize_annealers(self):
#         if self.log_writer is not None:
#             for annealed_var in self.annealed_vars:
#                 if rgetattr(self, annealed_var) is not None:
#                     self.log_writer.add_scalar('%s/%s' % (str(self.node_type), annealed_var.replace('.', '/')),
#                                                rgetattr(self, annealed_var), self.curr_iter)
#
#     def obtain_encoded_tensors(self,
#                                mode,
#                                inputs,
#                                inputs_st,
#                                labels,
#                                labels_st,
#                                first_history_indices,
#                                neighbors,
#                                neighbors_edge_value,
#                                robot,
#                                map) -> (torch.Tensor,
#                                         torch.Tensor,
#                                         torch.Tensor,
#                                         torch.Tensor,
#                                         torch.Tensor,
#                                         torch.Tensor):
#         """
#         Encodes input and output tensors for node and robot.
#
#         :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
#         :param inputs: Input tensor including the state for each agent over time [bs, t, state].
#         :param inputs_st: Standardized input tensor.
#         :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
#         :param labels_st: Standardized label tensor.
#         :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
#         :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
#                             [[bs, t, neighbor state]]
#         :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
#         :param robot: Standardized robot state over time. [bs, t, robot_state]
#         :param map: Tensor of Map information. [bs, channels, x, y]
#         :return: tuple(x, x_nr_t, y_e, y_r, y, n_s_t0)
#             WHERE
#             - x: Encoded input / condition tensor to the CVAE x_e.
#             - x_r_t: Robot state (if robot is in scene).
#             - y_e: Encoded label / future of the node.
#             - y_r: Encoded future of the robot.
#             - y: Label / future of the node.
#             - n_s_t0: Standardized current state of the node.
#         """
#
#         x, x_r_t, y_e, y_r, y = None, None, None, None, None
#         initial_dynamics = dict()
#
#         batch_size = inputs.shape[0]
#
#         #########################################
#         # Provide basic information to encoders #
#         #########################################
#         node_history = inputs
#         node_present_state = inputs[:, -1]
#         node_pos = inputs[:, -1, 0:2]
#         node_vel = inputs[:, -1, 2:4]
#
#         node_history_st = inputs_st
#         node_present_state_st = inputs_st[:, -1]
#         node_pos_st = inputs_st[:, -1, 0:2]
#         node_vel_st = inputs_st[:, -1, 2:4]
#
#         n_s_t0 = node_present_state_st
#
#         initial_dynamics['pos'] = node_pos
#         initial_dynamics['vel'] = node_vel
#
#         self.dynamic.set_initial_condition(initial_dynamics)
#
#         if self.hyperparams['incl_robot_node']:
#             x_r_t, y_r = robot[..., 0, :], robot[..., 1:, :]
#
#         ##################
#         # Encode History #
#         ##################
#         node_history_encoded = self.encode_node_history(mode,
#                                                         node_history_st,
#                                                         first_history_indices)
#
#         ##################
#         # Encode Present #
#         ##################
#         node_present = node_present_state_st  # [bs, state_dim]
#
#         ##################
#         # Encode Future #
#         ##################
#         if mode != ModeKeys.PREDICT:
#             y = labels_st
#
#         ##############################
#         # Encode Node Edges per Type #
#         ##############################
#         if self.hyperparams['edge_encoding']:
#             node_edges_encoded = list()
#             for edge_type in self.edge_types:
#                 # Encode edges for given edge type
#                 encoded_edges_type = self.encode_edge(mode,
#                                                       node_history,
#                                                       node_history_st,
#                                                       edge_type,
#                                                       neighbors[edge_type],
#                                                       neighbors_edge_value[edge_type],
#                                                       first_history_indices)
#                 node_edges_encoded.append(encoded_edges_type)  # List of [bs/nbs, enc_rnn_dim]
#             #####################
#             # Encode Node Edges #
#             #####################
#             total_edge_influence = self.encode_total_edge_influence(mode,
#                                                                     node_edges_encoded,
#                                                                     node_history_encoded,
#                                                                     batch_size)
#
#         ################
#         # Map Encoding #
#         ################
#         if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
#             if self.log_writer and (self.curr_iter + 1) % 500 == 0:
#                 map_clone = map.clone()
#                 map_patch = self.hyperparams['map_encoder'][self.node_type]['patch_size']
#                 map_clone[:, :, map_patch[1] - 5:map_patch[1] + 5, map_patch[0] - 5:map_patch[0] + 5] = 1.
#                 self.log_writer.add_images(f"{self.node_type}/cropped_maps", map_clone,
#                                            self.curr_iter, dataformats='NCWH')
#
#             encoded_map = self.node_modules[self.node_type + '/map_encoder'](map * 2. - 1., (mode == ModeKeys.TRAIN))
#             do = self.hyperparams['map_encoder'][self.node_type]['dropout']
#             encoded_map = F.dropout(encoded_map, do, training=(mode == ModeKeys.TRAIN))
#
#         ######################################
#         # Concatenate Encoder Outputs into x #
#         ######################################
#         x_concat_list = list()
#
#         # Every node has an edge-influence encoder (which could just be zero).
#         if self.hyperparams['edge_encoding']:
#             x_concat_list.append(total_edge_influence)  # [bs/nbs, 4*enc_rnn_dim]
#
#         # Every node has a history encoder.
#         x_concat_list.append(node_history_encoded)  # [bs/nbs, enc_rnn_dim_history]
#
#         if self.hyperparams['incl_robot_node']:
#             robot_future_encoder = self.encode_robot_future(mode, x_r_t, y_r)
#             x_concat_list.append(robot_future_encoder)
#
#         if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
#             if self.log_writer:
#                 self.log_writer.add_scalar(f"{self.node_type}/encoded_map_max",
#                                            torch.max(torch.abs(encoded_map)), self.curr_iter)
#             x_concat_list.append(encoded_map)
#
#         x = torch.cat(x_concat_list, dim=1)
#
#         if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
#             y_e = self.encode_node_future(mode, node_present, y)
#
#         return x, x_r_t, y_e, y_r, y, n_s_t0
#
#     def encode_node_history(self, mode, node_hist, first_history_indices):
#         """
#         Encodes the nodes history.
#
#         :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
#         :param node_hist: Historic and current state of the node. [bs, mhl, state]
#         :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
#         :return: Encoded node history tensor. [bs, enc_rnn_dim]
#         """
#         outputs, _ = run_lstm_on_variable_length_seqs(self.node_modules[self.node_type + '/node_history_encoder'],
#                                                       original_seqs=node_hist,
#                                                       lower_indices=first_history_indices)
#
#         outputs = F.dropout(outputs,
#                             p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
#                             training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]
#
#         last_index_per_sequence = -(first_history_indices + 1)
#
#         return outputs[torch.arange(first_history_indices.shape[0]), last_index_per_sequence]
#
#     def encode_edge(self,
#                     mode,
#                     node_history,
#                     node_history_st,
#                     edge_type,
#                     neighbors,
#                     neighbors_edge_value,
#                     first_history_indices):
#
#         max_hl = self.hyperparams['maximum_history_length']
#
#         edge_states_list = list()  # list of [#of neighbors, max_ht, state_dim]
#         for i, neighbor_states in enumerate(neighbors):  # Get neighbors for timestep in batch
#             if len(neighbor_states) == 0:  # There are no neighbors for edge type # TODO necessary?
#                 neighbor_state_length = int(
#                     np.sum([len(entity_dims) for entity_dims in self.state[edge_type[1]].values()])
#                 )
#                 edge_states_list.append(torch.zeros((1, max_hl + 1, neighbor_state_length), device=self.device))
#             else:
#                 edge_states_list.append(torch.stack(neighbor_states, dim=0).to(self.device))
#
#         if self.hyperparams['edge_state_combine_method'] == 'sum':
#             # Used in Structural-RNN to combine edges as well.
#             op_applied_edge_states_list = list()
#             for neighbors_state in edge_states_list:
#                 op_applied_edge_states_list.append(torch.sum(neighbors_state, dim=0))
#             combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)
#             if self.hyperparams['dynamic_edges'] == 'yes':
#                 # Should now be (bs, time, 1)
#                 op_applied_edge_mask_list = list()
#                 for edge_value in neighbors_edge_value:
#                     op_applied_edge_mask_list.append(torch.clamp(torch.sum(edge_value.to(self.device),
#                                                                            dim=0, keepdim=True), max=1.))
#                 combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0)
#
#         elif self.hyperparams['edge_state_combine_method'] == 'max':
#             # Used in NLP, e.g. max over word embeddings in a sentence.
#             op_applied_edge_states_list = list()
#             for neighbors_state in edge_states_list:
#                 op_applied_edge_states_list.append(torch.max(neighbors_state, dim=0))
#             combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)
#             if self.hyperparams['dynamic_edges'] == 'yes':
#                 # Should now be (bs, time, 1)
#                 op_applied_edge_mask_list = list()
#                 for edge_value in neighbors_edge_value:
#                     op_applied_edge_mask_list.append(torch.clamp(torch.max(edge_value.to(self.device),
#                                                                            dim=0, keepdim=True), max=1.))
#                 combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0)
#
#         elif self.hyperparams['edge_state_combine_method'] == 'mean':
#             # Used in NLP, e.g. mean over word embeddings in a sentence.
#             op_applied_edge_states_list = list()
#             for neighbors_state in edge_states_list:
#                 op_applied_edge_states_list.append(torch.mean(neighbors_state, dim=0))
#             combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)
#             if self.hyperparams['dynamic_edges'] == 'yes':
#                 # Should now be (bs, time, 1)
#                 op_applied_edge_mask_list = list()
#                 for edge_value in neighbors_edge_value:
#                     op_applied_edge_mask_list.append(torch.clamp(torch.mean(edge_value.to(self.device),
#                                                                             dim=0, keepdim=True), max=1.))
#                 combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0)
#
#         joint_history = torch.cat([combined_neighbors, node_history_st], dim=-1)
#
#         outputs, _ = run_lstm_on_variable_length_seqs(
#             self.node_modules[DirectedEdge.get_str_from_types(*edge_type) + '/edge_encoder'],
#             original_seqs=joint_history,
#             lower_indices=first_history_indices
#         )
#
#         outputs = F.dropout(outputs,
#                             p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
#                             training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]
#
#         last_index_per_sequence = -(first_history_indices + 1)
#         ret = outputs[torch.arange(last_index_per_sequence.shape[0]), last_index_per_sequence]
#         if self.hyperparams['dynamic_edges'] == 'yes':
#             return ret * combined_edge_masks
#         else:
#             return ret
#
#     def encode_total_edge_influence(self, mode, encoded_edges, node_history_encoder, batch_size):
#         if self.hyperparams['edge_influence_combine_method'] == 'sum':
#             stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
#             combined_edges = torch.sum(stacked_encoded_edges, dim=0)
#
#         elif self.hyperparams['edge_influence_combine_method'] == 'mean':
#             stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
#             combined_edges = torch.mean(stacked_encoded_edges, dim=0)
#
#         elif self.hyperparams['edge_influence_combine_method'] == 'max':
#             stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
#             combined_edges = torch.max(stacked_encoded_edges, dim=0)
#
#         elif self.hyperparams['edge_influence_combine_method'] == 'bi-rnn':
#             if len(encoded_edges) == 0:
#                 combined_edges = torch.zeros((batch_size, self.eie_output_dims), device=self.device)
#
#             else:
#                 # axis=1 because then we get size [batch_size, max_time, depth]
#                 encoded_edges = torch.stack(encoded_edges, dim=1)
#
#                 _, state = self.node_modules[self.node_type + '/edge_influence_encoder'](encoded_edges)
#                 combined_edges = unpack_RNN_state(state)
#                 combined_edges = F.dropout(combined_edges,
#                                            p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
#                                            training=(mode == ModeKeys.TRAIN))
#
#         elif self.hyperparams['edge_influence_combine_method'] == 'attention':
#             # Used in Social Attention (https://arxiv.org/abs/1710.04689)
#             if len(encoded_edges) == 0:
#                 combined_edges = torch.zeros((batch_size, self.eie_output_dims), device=self.device)
#
#             else:
#                 # axis=1 because then we get size [batch_size, max_time, depth]
#                 encoded_edges = torch.stack(encoded_edges, dim=1)
#                 combined_edges, _ = self.node_modules[self.node_type + '/edge_influence_encoder'](encoded_edges,
#                                                                                                   node_history_encoder)
#                 combined_edges = F.dropout(combined_edges,
#                                            p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
#                                            training=(mode == ModeKeys.TRAIN))
#
#         return combined_edges
#
#     def encode_node_future(self, mode, node_present, node_future) -> torch.Tensor:
#         """
#         Encodes the node future (during training) using a bi-directional LSTM
#
#         :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
#         :param node_present: Current state of the node. [bs, state]
#         :param node_future: Future states of the node. [bs, ph, state]
#         :return: Encoded future.
#         """
#         initial_h_model = self.node_modules[self.node_type + '/node_future_encoder/initial_h']
#         initial_c_model = self.node_modules[self.node_type + '/node_future_encoder/initial_c']
#
#         # Here we're initializing the forward hidden states,
#         # but zeroing the backward ones.
#         initial_h = initial_h_model(node_present)
#         initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0)
#
#         initial_c = initial_c_model(node_present)
#         initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)
#
#         initial_state = (initial_h, initial_c)
#
#         _, state = self.node_modules[self.node_type + '/node_future_encoder'](node_future, initial_state)
#         state = unpack_RNN_state(state)
#         state = F.dropout(state,
#                           p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
#                           training=(mode == ModeKeys.TRAIN))
#
#         return state
#
#     def encode_robot_future(self, mode, robot_present, robot_future) -> torch.Tensor:
#         """
#         Encodes the robot future (during training) using a bi-directional LSTM
#
#         :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
#         :param robot_present: Current state of the robot. [bs, state]
#         :param robot_future: Future states of the robot. [bs, ph, state]
#         :return: Encoded future.
#         """
#         initial_h_model = self.node_modules['robot_future_encoder/initial_h']
#         initial_c_model = self.node_modules['robot_future_encoder/initial_c']
#
#         # Here we're initializing the forward hidden states,
#         # but zeroing the backward ones.
#         initial_h = initial_h_model(robot_present)
#         initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0)
#
#         initial_c = initial_c_model(robot_present)
#         initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)
#
#         initial_state = (initial_h, initial_c)
#
#         _, state = self.node_modules['robot_future_encoder'](robot_future, initial_state)
#         state = unpack_RNN_state(state)
#         state = F.dropout(state,
#                           p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
#                           training=(mode == ModeKeys.TRAIN))
#
#         return state
#
#     def q_z_xy(self, mode, x, y_e) -> torch.Tensor:
#         r"""
#         .. math:: q_\phi(z \mid \mathbf{x}_i, \mathbf{y}_i)
#
#         :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
#         :param x: Input / Condition tensor.
#         :param y_e: Encoded future tensor.
#         :return: Latent distribution of the CVAE.
#         """
#         xy = torch.cat([x, y_e], dim=1)
#
#         if self.hyperparams['q_z_xy_MLP_dims'] is not None:
#             dense = self.node_modules[self.node_type + '/q_z_xy']
#             h = F.dropout(F.relu(dense(xy)),
#                           p=1. - self.hyperparams['MLP_dropout_keep_prob'],
#                           training=(mode == ModeKeys.TRAIN))
#
#         else:
#             h = xy
#
#         to_latent = self.node_modules[self.node_type + '/hxy_to_z']
#         return self.latent.dist_from_h(to_latent(h), mode)
#
#     def p_z_x(self, mode, x):
#         r"""
#         .. math:: p_\theta(z \mid \mathbf{x}_i)
#
#         :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
#         :param x: Input / Condition tensor.
#         :return: Latent distribution of the CVAE.
#         """
#         if self.hyperparams['p_z_x_MLP_dims'] is not None:
#             dense = self.node_modules[self.node_type + '/p_z_x']
#             h = F.dropout(F.relu(dense(x)),
#                           p=1. - self.hyperparams['MLP_dropout_keep_prob'],
#                           training=(mode == ModeKeys.TRAIN))
#
#         else:
#             h = x
#
#         to_latent = self.node_modules[self.node_type + '/hx_to_z']
#         return self.latent.dist_from_h(to_latent(h), mode)
#
#     def project_to_GMM_params(self, tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
#         """
#         Projects tensor to parameters of a GMM with N components and D dimensions.
#
#         :param tensor: Input tensor.
#         :return: tuple(log_pis, mus, log_sigmas, corrs)
#             WHERE
#             - log_pis: Weight (logarithm) of each GMM component. [N]
#             - mus: Mean of each GMM component. [N, D]
#             - log_sigmas: Standard Deviation (logarithm) of each GMM component. [N, D]
#             - corrs: Correlation between the GMM components. [N]
#         """
#         log_pis = self.node_modules[self.node_type + '/decoder/proj_to_GMM_log_pis'](tensor)
#         mus = self.node_modules[self.node_type + '/decoder/proj_to_GMM_mus'](tensor)
#         log_sigmas = self.node_modules[self.node_type + '/decoder/proj_to_GMM_log_sigmas'](tensor)
#         corrs = torch.tanh(self.node_modules[self.node_type + '/decoder/proj_to_GMM_corrs'](tensor))
#         return log_pis, mus, log_sigmas, corrs
#
#     def p_y_xz(self, mode, x, x_nr_t, y_r, n_s_t0, z_stacked, prediction_horizon,
#                num_samples, num_components=1, gmm_mode=False):
#         r"""
#         .. math:: p_\psi(\mathbf{y}_i \mid \mathbf{x}_i, z)
#
#         :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
#         :param x: Input / Condition tensor.
#         :param x_nr_t: Joint state of node and robot (if robot is in scene).
#         :param y: Future tensor.
#         :param y_r: Encoded future tensor.
#         :param n_s_t0: Standardized current state of the node.
#         :param z_stacked: Stacked latent state. [num_samples_z * num_samples_gmm, bs, latent_state]
#         :param prediction_horizon: Number of prediction timesteps.
#         :param num_samples: Number of samples from the latent space.
#         :param num_components: Number of GMM components.
#         :param gmm_mode: If True: The mode of the GMM is sampled.
#         :return: GMM2D. If mode is Predict, also samples from the GMM.
#         """
#         ph = prediction_horizon
#         pred_dim = self.pred_state_length
#
#         z = torch.reshape(z_stacked, (-1, self.latent.z_dim))
#         zx = torch.cat([z, x.repeat(num_samples * num_components, 1)], dim=1)
#
#         cell = self.node_modules[self.node_type + '/decoder/rnn_cell']
#         initial_h_model = self.node_modules[self.node_type + '/decoder/initial_h']
#
#         initial_state = initial_h_model(zx)
#
#         log_pis, mus, log_sigmas, corrs, a_sample = [], [], [], [], []
#
#         # Infer initial action state for node from current state
#         a_0 = self.node_modules[self.node_type + '/decoder/state_action'](n_s_t0)
#
#         state = initial_state
#         if self.hyperparams['incl_robot_node']:
#             input_ = torch.cat([zx,
#                                 a_0.repeat(num_samples * num_components, 1),
#                                 x_nr_t.repeat(num_samples * num_components, 1)], dim=1)
#         else:
#             input_ = torch.cat([zx, a_0.repeat(num_samples * num_components, 1)], dim=1)
#
#         for j in range(ph):
#             h_state = cell(input_, state)
#             log_pi_t, mu_t, log_sigma_t, corr_t = self.project_to_GMM_params(h_state)
#
#             gmm = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t)  # [k;bs, pred_dim]
#
#             if mode == ModeKeys.PREDICT and gmm_mode:
#                 a_t = gmm.mode()
#             else:
#                 a_t = gmm.rsample()
#
#             if num_components > 1:
#                 if mode == ModeKeys.PREDICT:
#                     log_pis.append(self.latent.p_dist.logits.repeat(num_samples, 1, 1))
#                 else:
#                     log_pis.append(self.latent.q_dist.logits.repeat(num_samples, 1, 1))
#             else:
#                 log_pis.append(
#                     torch.ones_like(corr_t.reshape(num_samples, num_components, -1).permute(0, 2, 1).reshape(-1, 1))
#                 )
#
#             mus.append(
#                 mu_t.reshape(
#                     num_samples, num_components, -1, 2
#                 ).permute(0, 2, 1, 3).reshape(-1, 2 * num_components)
#             )
#             log_sigmas.append(
#                 log_sigma_t.reshape(
#                     num_samples, num_components, -1, 2
#                 ).permute(0, 2, 1, 3).reshape(-1, 2 * num_components))
#             corrs.append(
#                 corr_t.reshape(
#                     num_samples, num_components, -1
#                 ).permute(0, 2, 1).reshape(-1, num_components))
#
#             if self.hyperparams['incl_robot_node']:
#                 dec_inputs = [zx, a_t, y_r[:, j].repeat(num_samples * num_components, 1)]
#             else:
#                 dec_inputs = [zx, a_t]
#             input_ = torch.cat(dec_inputs, dim=1)
#             state = h_state
#
#         log_pis = torch.stack(log_pis, dim=1)
#         mus = torch.stack(mus, dim=1)
#         log_sigmas = torch.stack(log_sigmas, dim=1)
#         corrs = torch.stack(corrs, dim=1)
#
#         a_dist = GMM2D(torch.reshape(log_pis, [num_samples, -1, ph, num_components]),
#                        torch.reshape(mus, [num_samples, -1, ph, num_components * pred_dim]),
#                        torch.reshape(log_sigmas, [num_samples, -1, ph, num_components * pred_dim]),
#                        torch.reshape(corrs, [num_samples, -1, ph, num_components]))
#
#         if self.hyperparams['dynamic'][self.node_type]['distribution']:
#             y_dist = self.dynamic.integrate_distribution(a_dist, x)
#         else:
#             y_dist = a_dist
#
#         if mode == ModeKeys.PREDICT:
#             if gmm_mode:
#                 a_sample = a_dist.mode()
#             else:
#                 a_sample = a_dist.rsample()
#             sampled_future = self.dynamic.integrate_samples(a_sample, x)
#             return y_dist, sampled_future
#         else:
#             return y_dist
#
#     def encoder(self, mode, x, y_e, num_samples=None):
#         """
#         Encoder of the CVAE.
#
#         :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
#         :param x: Input / Condition tensor.
#         :param y_e: Encoded future tensor.
#         :param num_samples: Number of samples from the latent space during Prediction.
#         :return: tuple(z, kl_obj)
#             WHERE
#             - z: Samples from the latent space.
#             - kl_obj: KL Divergenze between q and p
#         """
#         if mode == ModeKeys.TRAIN:
#             sample_ct = self.hyperparams['k']
#         elif mode == ModeKeys.EVAL:
#             sample_ct = self.hyperparams['k_eval']
#         elif mode == ModeKeys.PREDICT:
#             sample_ct = num_samples
#             if num_samples is None:
#                 raise ValueError("num_samples cannot be None with mode == PREDICT.")
#
#         self.latent.q_dist = self.q_z_xy(mode, x, y_e)
#         self.latent.p_dist = self.p_z_x(mode, x)
#
#         z = self.latent.sample_q(sample_ct, mode)
#
#         if mode == ModeKeys.TRAIN:
#             kl_obj = self.latent.kl_q_p(self.log_writer, '%s' % str(self.node_type), self.curr_iter)
#             if self.log_writer is not None:
#                 self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'kl'), kl_obj, self.curr_iter)
#         else:
#             kl_obj = None
#
#         return z, kl_obj
#
#     def decoder(self, mode, x, x_nr_t, y, y_r, n_s_t0, z, labels, prediction_horizon, num_samples):
#         """
#         Decoder of the CVAE.
#
#         :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
#         :param x: Input / Condition tensor.
#         :param x: Input / Condition tensor.
#         :param x_nr_t: Joint state of node and robot (if robot is in scene).
#         :param y: Future tensor.
#         :param y_r: Encoded future tensor.
#         :param n_s_t0: Standardized current state of the node.
#         :param z: Stacked latent state.
#         :param prediction_horizon: Number of prediction timesteps.
#         :param num_samples: Number of samples from the latent space.
#         :return: Log probability of y over p.
#         """
#
#         num_components = self.hyperparams['N'] * self.hyperparams['K']
#         y_dist = self.p_y_xz(mode, x, x_nr_t, y_r, n_s_t0, z,
#                              prediction_horizon, num_samples, num_components=num_components)
#         log_p_yt_xz = torch.clamp(y_dist.log_prob(labels), max=self.hyperparams['log_p_yt_xz_max'])
#         if self.hyperparams['log_histograms'] and self.log_writer is not None:
#             self.log_writer.add_histogram('%s/%s' % (str(self.node_type), 'log_p_yt_xz'), log_p_yt_xz, self.curr_iter)
#
#         log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)
#         return log_p_y_xz
#
#     def train_loss(self,
#                    inputs,
#                    inputs_st,
#                    first_history_indices,
#                    labels,
#                    labels_st,
#                    neighbors,
#                    neighbors_edge_value,
#                    robot,
#                    map,
#                    prediction_horizon) -> torch.Tensor:
#         """
#         Calculates the training loss for a batch.
#
#         :param inputs: Input tensor including the state for each agent over time [bs, t, state].
#         :param inputs_st: Standardized input tensor.
#         :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
#         :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
#         :param labels_st: Standardized label tensor.
#         :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
#                             [[bs, t, neighbor state]]
#         :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
#         :param robot: Standardized robot state over time. [bs, t, robot_state]
#         :param map: Tensor of Map information. [bs, channels, x, y]
#         :param prediction_horizon: Number of prediction timesteps.
#         :return: Scalar tensor -> nll loss
#         """
#         mode = ModeKeys.TRAIN
#
#         x, x_nr_t, y_e, y_r, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
#                                                                      inputs=inputs,
#                                                                      inputs_st=inputs_st,
#                                                                      labels=labels,
#                                                                      labels_st=labels_st,
#                                                                      first_history_indices=first_history_indices,
#                                                                      neighbors=neighbors,
#                                                                      neighbors_edge_value=neighbors_edge_value,
#                                                                      robot=robot,
#                                                                      map=map)
#
#         z, kl = self.encoder(mode, x, y_e)
#         log_p_y_xz = self.decoder(mode, x, x_nr_t, y, y_r, n_s_t0, z,
#                                   labels,  # Loss is calculated on unstandardized label
#                                   prediction_horizon,
#                                   self.hyperparams['k'])
#
#         log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]
#         log_likelihood = torch.mean(log_p_y_xz_mean)
#
#         mutual_inf_q = mutual_inf_mc(self.latent.q_dist)
#         mutual_inf_p = mutual_inf_mc(self.latent.p_dist)
#
#         ELBO = log_likelihood - self.kl_weight * kl + 1. * mutual_inf_p
#         loss = -ELBO
#
#         if self.hyperparams['log_histograms'] and self.log_writer is not None:
#             self.log_writer.add_histogram('%s/%s' % (str(self.node_type), 'log_p_y_xz'),
#                                           log_p_y_xz_mean,
#                                           self.curr_iter)
#
#         if self.log_writer is not None:
#             self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'mutual_information_q'),
#                                        mutual_inf_q,
#                                        self.curr_iter)
#             self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'mutual_information_p'),
#                                        mutual_inf_p,
#                                        self.curr_iter)
#             self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'log_likelihood'),
#                                        log_likelihood,
#                                        self.curr_iter)
#             self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'loss'),
#                                        loss,
#                                        self.curr_iter)
#             if self.hyperparams['log_histograms']:
#                 self.latent.summarize_for_tensorboard(self.log_writer, str(self.node_type), self.curr_iter)
#         return loss
#
#     def eval_loss(self,
#                   inputs,
#                   inputs_st,
#                   first_history_indices,
#                   labels,
#                   labels_st,
#                   neighbors,
#                   neighbors_edge_value,
#                   robot,
#                   map,
#                   prediction_horizon) -> torch.Tensor:
#         """
#         Calculates the evaluation loss for a batch.
#
#         :param inputs: Input tensor including the state for each agent over time [bs, t, state].
#         :param inputs_st: Standardized input tensor.
#         :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
#         :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
#         :param labels_st: Standardized label tensor.
#         :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
#                             [[bs, t, neighbor state]]
#         :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
#         :param robot: Standardized robot state over time. [bs, t, robot_state]
#         :param map: Tensor of Map information. [bs, channels, x, y]
#         :param prediction_horizon: Number of prediction timesteps.
#         :return: tuple(nll_q_is, nll_p, nll_exact, nll_sampled)
#         """
#
#         mode = ModeKeys.EVAL
#
#         x, x_nr_t, y_e, y_r, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
#                                                                      inputs=inputs,
#                                                                      inputs_st=inputs_st,
#                                                                      labels=labels,
#                                                                      labels_st=labels_st,
#                                                                      first_history_indices=first_history_indices,
#                                                                      neighbors=neighbors,
#                                                                      neighbors_edge_value=neighbors_edge_value,
#                                                                      robot=robot,
#                                                                      map=map)
#
#         num_components = self.hyperparams['N'] * self.hyperparams['K']
#         ### Importance sampled NLL estimate
#         z, _ = self.encoder(mode, x, y_e)  # [k_eval, nbs, N*K]
#         z = self.latent.sample_p(1, mode, full_dist=True)
#         y_dist, _ = self.p_y_xz(ModeKeys.PREDICT, x, x_nr_t, y_r, n_s_t0, z,
#                                 prediction_horizon, num_samples=1, num_components=num_components)
#         # We use unstandardized labels to compute the loss
#         log_p_yt_xz = torch.clamp(y_dist.log_prob(labels), max=self.hyperparams['log_p_yt_xz_max'])
#         log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)
#         log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]
#         log_likelihood = torch.mean(log_p_y_xz_mean)
#         nll = -log_likelihood
#
#         return nll
#
#     def predict(self,
#                 inputs,
#                 inputs_st,
#                 first_history_indices,
#                 neighbors,
#                 neighbors_edge_value,
#                 robot,
#                 map,
#                 prediction_horizon,
#                 num_samples,
#                 z_mode=False,
#                 gmm_mode=False,
#                 full_dist=True,
#                 all_z_sep=False):
#         """
#         Predicts the future of a batch of nodes.
#
#         :param inputs: Input tensor including the state for each agent over time [bs, t, state].
#         :param inputs_st: Standardized input tensor.
#         :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
#         :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
#                             [[bs, t, neighbor state]]
#         :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
#         :param robot: Standardized robot state over time. [bs, t, robot_state]
#         :param map: Tensor of Map information. [bs, channels, x, y]
#         :param prediction_horizon: Number of prediction timesteps.
#         :param num_samples: Number of samples from the latent space.
#         :param z_mode: If True: Select the most likely latent state.
#         :param gmm_mode: If True: The mode of the GMM is sampled.
#         :param all_z_sep: Samples each latent mode individually without merging them into a GMM.
#         :param full_dist: Samples all latent states and merges them into a GMM as output.
#         :return:
#         """
#         mode = ModeKeys.PREDICT
#
#         x, x_nr_t, _, y_r, _, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
#                                                                    inputs=inputs,
#                                                                    inputs_st=inputs_st,
#                                                                    labels=None,
#                                                                    labels_st=None,
#                                                                    first_history_indices=first_history_indices,
#                                                                    neighbors=neighbors,
#                                                                    neighbors_edge_value=neighbors_edge_value,
#                                                                    robot=robot,
#                                                                    map=map)
#
#         self.latent.p_dist = self.p_z_x(mode, x)
#         z, num_samples, num_components = self.latent.sample_p(num_samples,
#                                                               mode,
#                                                               most_likely_z=z_mode,
#                                                               full_dist=full_dist,
#                                                               all_z_sep=all_z_sep)
#
#         _, our_sampled_future = self.p_y_xz(mode, x, x_nr_t, y_r, n_s_t0, z,
#                                             prediction_horizon,
#                                             num_samples,
#                                             num_components,
#                                             gmm_mode)
#
#         return our_sampled_future



#######################################################################################################################
import warnings
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from ..model.components import *
# from ..model.model_utils import *
# from ..model.dynamics import *
# from ..environment.scene_graph import DirectedEdge
import torch
import torch.distributions as td
import numpy as np
from enum import Enum
import functools
import torch.nn.utils.rnn as rnn
import math

class Edge(object):
    def __init__(self, curr_node, other_node):
        self.id = self.get_edge_id(curr_node, other_node)
        self.type = self.get_edge_type(curr_node, other_node)
        self.curr_node = curr_node
        self.other_node = other_node

    @staticmethod
    def get_edge_id(n1, n2):
        raise NotImplementedError("Use one of the Edge subclasses!")

    @staticmethod
    def get_str_from_types(nt1, nt2):
        raise NotImplementedError("Use one of the Edge subclasses!")

    @staticmethod
    def get_edge_type(n1, n2):
        raise NotImplementedError("Use one of the Edge subclasses!")

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.id == other.id)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return self.id
class DirectedEdge(Edge):
    def __init__(self, curr_node, other_node):
        super(DirectedEdge, self).__init__(curr_node, other_node)

    @staticmethod
    def get_edge_id(n1, n2):
        return '->'.join([str(n1), str(n2)])

    @staticmethod
    def get_str_from_types(nt1, nt2):
        return '->'.join([nt1.name, nt2.name])

    @staticmethod
    def get_edge_type(n1, n2):
        return '->'.join([n1.type.name, n2.type.name])
def to_one_hot(labels, n_labels):
    return torch.eye(n_labels, device=labels.device)[labels]


class GMM2D(td.Distribution):
    r"""
    Gaussian Mixture Model using 2D Multivariate Gaussians each of as N components:
    Cholesky decompesition and affine transformation for sampling:

    .. math:: Z \sim N(0, I)

    .. math:: S = \mu + LZ

    .. math:: S \sim N(\mu, \Sigma) \rightarrow N(\mu, LL^T)

    where :math:`L = chol(\Sigma)` and

    .. math:: \Sigma = \left[ {\begin{array}{cc} \sigma^2_x & \rho \sigma_x \sigma_y \\ \rho \sigma_x \sigma_y & \sigma^2_y \\ \end{array} } \right]

    such that

    .. math:: L = chol(\Sigma) = \left[ {\begin{array}{cc} \sigma_x & 0 \\ \rho \sigma_y & \sigma_y \sqrt{1-\rho^2} \\ \end{array} } \right]

    :param log_pis: Log Mixing Proportions :math:`log(\pi)`. [..., N]
    :param mus: Mixture Components mean :math:`\mu`. [..., N * 2]
    :param log_sigmas: Log Standard Deviations :math:`log(\sigma_d)`. [..., N * 2]
    :param corrs: Cholesky factor of correlation :math:`\rho`. [..., N]
    :param clip_lo: Clips the lower end of the standard deviation.
    :param clip_hi: Clips the upper end of the standard deviation.
    """

    def __init__(self, log_pis, mus, log_sigmas, corrs):
        super(GMM2D, self).__init__(
            batch_shape=log_pis.shape[0], event_shape=log_pis.shape[1:]
        )
        self.components = log_pis.shape[-1]
        self.dimensions = 2
        self.device = log_pis.device

        log_pis = torch.clamp(log_pis, min=-1e5)
        self.log_pis = log_pis - torch.logsumexp(
            log_pis, dim=-1, keepdim=True
        )  # [..., N]
        self.mus = self.reshape_to_components(mus)  # [..., N, 2]
        self.log_sigmas = self.reshape_to_components(log_sigmas)  # [..., N, 2]
        self.sigmas = torch.exp(self.log_sigmas)  # [..., N, 2]
        self.one_minus_rho2 = 1 - corrs**2  # [..., N]
        self.one_minus_rho2 = torch.clamp(
            self.one_minus_rho2, min=1e-5, max=1
        )  # otherwise log can be nan
        self.corrs = corrs  # [..., N]

        self.L = torch.stack(
            [
                torch.stack(
                    [self.sigmas[..., 0], torch.zeros_like(self.log_pis)], dim=-1
                ),
                torch.stack(
                    [
                        self.sigmas[..., 1] * self.corrs,
                        self.sigmas[..., 1] * torch.sqrt(self.one_minus_rho2),
                    ],
                    dim=-1,
                ),
            ],
            dim=-2,
        )

        self.pis_cat_dist = td.Categorical(logits=log_pis)

    @classmethod
    def from_log_pis_mus_cov_mats(cls, log_pis, mus, cov_mats):
        corrs_sigma12 = cov_mats[..., 0, 1]
        sigma_1 = torch.clamp(cov_mats[..., 0, 0], min=1e-8)
        sigma_2 = torch.clamp(cov_mats[..., 1, 1], min=1e-8)
        sigmas = torch.stack([torch.sqrt(sigma_1), torch.sqrt(sigma_2)], dim=-1)
        log_sigmas = torch.log(sigmas)
        corrs = corrs_sigma12 / (torch.prod(sigmas, dim=-1))
        return cls(log_pis, mus, log_sigmas, corrs)

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.

        :param sample_shape: Shape of the samples
        :return: Samples from the GMM.
        """
        mvn_samples = self.mus + torch.squeeze(
            torch.matmul(
                self.L,
                torch.unsqueeze(
                    torch.randn(size=sample_shape + self.mus.shape, device=self.device),
                    dim=-1,
                ),
            ),
            dim=-1,
        )
        component_cat_samples = self.pis_cat_dist.sample(sample_shape)
        selector = torch.unsqueeze(
            to_one_hot(component_cat_samples, self.components), dim=-1
        )
        return torch.sum(mvn_samples * selector, dim=-2)

    def log_prob(self, value):
        r"""
        Calculates the log probability of a value using the PDF for bivariate normal distributions:

        .. math::
            f(x | \mu, \sigma, \rho)={\frac {1}{2\pi \sigma _{x}\sigma _{y}{\sqrt {1-\rho ^{2}}}}}\exp
            \left(-{\frac {1}{2(1-\rho ^{2})}}\left[{\frac {(x-\mu _{x})^{2}}{\sigma _{x}^{2}}}+
            {\frac {(y-\mu _{y})^{2}}{\sigma _{y}^{2}}}-{\frac {2\rho (x-\mu _{x})(y-\mu _{y})}
            {\sigma _{x}\sigma _{y}}}\right]\right)

        :param value: The log probability density function is evaluated at those values.
        :return: Log probability
        """
        # x: [..., 2]
        value = torch.unsqueeze(value, dim=-2)  # [..., 1, 2]
        dx = value - self.mus  # [..., N, 2]

        exp_nominator = torch.sum(
            (dx / self.sigmas) ** 2, dim=-1
        ) - 2 * self.corrs * torch.prod(  # first and second term of exp nominator
            dx, dim=-1
        ) / torch.prod(
            self.sigmas, dim=-1
        )  # [..., N]

        component_log_p = (
            -(
                2 * np.log(2 * np.pi)
                + torch.log(self.one_minus_rho2)
                + 2 * torch.sum(self.log_sigmas, dim=-1)
                + exp_nominator / self.one_minus_rho2
            )
            / 2
        )

        return torch.logsumexp(self.log_pis + component_log_p, dim=-1)

    def get_for_node_at_time(self, n, t):
        return self.__class__(
            self.log_pis[:, n : n + 1, t : t + 1],
            self.mus[:, n : n + 1, t : t + 1],
            self.log_sigmas[:, n : n + 1, t : t + 1],
            self.corrs[:, n : n + 1, t : t + 1],
        )

    def mode(self):
        """
        Calculates the mode of the GMM by calculating probabilities of a 2D mesh grid

        :param required_accuracy: Accuracy of the meshgrid
        :return: Mode of the GMM
        """
        if self.mus.shape[-2] > 1:
            samp, bs, time, comp, _ = self.mus.shape
            assert samp == 1, "For taking the mode only one sample makes sense."
            mode_node_list = []
            for n in range(bs):
                mode_t_list = []
                for t in range(time):
                    nt_gmm = self.get_for_node_at_time(n, t)
                    x_min = self.mus[:, n, t, :, 0].min()
                    x_max = self.mus[:, n, t, :, 0].max()
                    y_min = self.mus[:, n, t, :, 1].min()
                    y_max = self.mus[:, n, t, :, 1].max()
                    search_grid = (
                        torch.stack(
                            torch.meshgrid(
                                [
                                    torch.arange(x_min, x_max, 0.01),
                                    torch.arange(y_min, y_max, 0.01),
                                ]
                            ),
                            dim=2,
                        )
                        .view(-1, 2)
                        .float()
                        .to(self.device)
                    )

                    ll_score = nt_gmm.log_prob(search_grid)
                    argmax = torch.argmax(ll_score.squeeze(), dim=0)
                    mode_t_list.append(search_grid[argmax])
                mode_node_list.append(torch.stack(mode_t_list, dim=0))
            return torch.stack(mode_node_list, dim=0).unsqueeze(dim=0)
        return torch.squeeze(self.mus, dim=-2)

    def reshape_to_components(self, tensor):
        if len(tensor.shape) == 5:
            return tensor
        return torch.reshape(
            tensor, list(tensor.shape[:-1]) + [self.components, self.dimensions]
        )

    def get_covariance_matrix(self):
        cov = self.corrs * torch.prod(self.sigmas, dim=-1)
        E = torch.stack(
            [
                torch.stack([self.sigmas[..., 0] ** 2, cov], dim=-1),
                torch.stack([cov, self.sigmas[..., 1] ** 2], dim=-1),
            ],
            dim=-2,
        )
        return E


class ModeKeys(Enum):
    TRAIN = 1
    EVAL = 2
    PREDICT = 3


class DiscreteLatent(object):
    def __init__(self, hyperparams, device):
        self.hyperparams = hyperparams
        self.z_dim = hyperparams["N"] * hyperparams["K"]
        self.N = hyperparams["N"]
        self.K = hyperparams["K"]
        self.kl_min = hyperparams["kl_min"]
        self.device = device
        self.temp = None  # filled in by MultimodalGenerativeCVAE.set_annealing_params
        self.z_logit_clip = (
            None  # filled in by MultimodalGenerativeCVAE.set_annealing_params
        )
        self.p_dist = None  # filled in by MultimodalGenerativeCVAE.encoder
        self.q_dist = None  # filled in by MultimodalGenerativeCVAE.encoder

    def dist_from_h(self, h, mode):
        logits_separated = torch.reshape(h, (-1, self.N, self.K))
        logits_separated_mean_zero = logits_separated - torch.mean(
            logits_separated, dim=-1, keepdim=True
        )
        if self.z_logit_clip is not None and mode == ModeKeys.TRAIN:
            c = self.z_logit_clip
            logits = torch.clamp(logits_separated_mean_zero, min=-c, max=c)
        else:
            logits = logits_separated_mean_zero

        return td.OneHotCategorical(logits=logits)

    def sample_q(self, num_samples, mode):
        bs = self.p_dist.probs.size()[0]
        num_components = self.N * self.K
        z_NK = (
            torch.from_numpy(self.all_one_hot_combinations(self.N, self.K))
            .float()
            .to(self.device)
            .repeat(num_samples, bs)
        )
        return torch.reshape(z_NK, (num_samples * num_components, -1, self.z_dim))

    def sample_p(
        self, num_samples, mode, most_likely_z=False, full_dist=True, all_z_sep=False
    ):
        num_components = 1
        if full_dist:
            bs = self.p_dist.probs.size()[0]
            z_NK = (
                torch.from_numpy(self.all_one_hot_combinations(self.N, self.K))
                .float()
                .to(self.device)
                .repeat(num_samples, bs)
            )
            num_components = self.K**self.N
            k = num_samples * num_components
        elif all_z_sep:
            bs = self.p_dist.probs.size()[0]
            z_NK = (
                torch.from_numpy(self.all_one_hot_combinations(self.N, self.K))
                .float()
                .to(self.device)
                .repeat(1, bs)
            )
            k = self.K**self.N
            num_samples = k
        elif most_likely_z:
            # Sampling the most likely z from p(z|x).
            eye_mat = torch.eye(self.p_dist.event_shape[-1], device=self.device)
            argmax_idxs = torch.argmax(self.p_dist.probs, dim=2)
            z_NK = torch.unsqueeze(eye_mat[argmax_idxs], dim=0).expand(
                num_samples, -1, -1, -1
            )
            k = num_samples
        else:
            z_NK = self.p_dist.sample((num_samples,))
            k = num_samples

        if mode == ModeKeys.PREDICT:
            return (
                torch.reshape(z_NK, (k, -1, self.N * self.K)),
                num_samples,
                num_components,
            )
        else:
            return torch.reshape(z_NK, (k, -1, self.N * self.K))

    def kl_q_p(self, log_writer=None, prefix=None, curr_iter=None):
        kl_separated = td.kl_divergence(self.q_dist, self.p_dist)
        if len(kl_separated.size()) < 2:
            kl_separated = torch.unsqueeze(kl_separated, dim=0)

        kl_minibatch = torch.mean(kl_separated, dim=0, keepdim=True)

        if log_writer is not None:
            log_writer.add_scalar(
                prefix + "/true_kl", torch.sum(kl_minibatch), curr_iter
            )

        if self.kl_min > 0:
            kl_lower_bounded = torch.clamp(kl_minibatch, min=self.kl_min)
            kl = torch.sum(kl_lower_bounded)
        else:
            kl = torch.sum(kl_minibatch)

        return kl

    def q_log_prob(self, z):
        k = z.size()[0]
        z_NK = torch.reshape(z, [k, -1, self.N, self.K])
        return torch.sum(self.q_dist.log_prob(z_NK), dim=2)

    def p_log_prob(self, z):
        k = z.size()[0]
        z_NK = torch.reshape(z, [k, -1, self.N, self.K])
        return torch.sum(self.p_dist.log_prob(z_NK), dim=2)

    def get_p_dist_probs(self):
        return self.p_dist.probs

    @staticmethod
    def all_one_hot_combinations(N, K):
        return (
            np.eye(K)
            .take(np.reshape(np.indices([K] * N), [N, -1]).T, axis=0)
            .reshape(-1, N * K)
        )  # [K**N, N*K]

    def summarize_for_tensorboard(self, log_writer, prefix, curr_iter):
        log_writer.add_histogram(prefix + "/latent/p_z_x", self.p_dist.probs, curr_iter)
        log_writer.add_histogram(
            prefix + "/latent/q_z_xy", self.q_dist.probs, curr_iter
        )
        log_writer.add_histogram(
            prefix + "/latent/p_z_x_logits", self.p_dist.logits, curr_iter
        )
        log_writer.add_histogram(
            prefix + "/latent/q_z_xy_logits", self.q_dist.logits, curr_iter
        )
        if self.z_dim <= 9:
            for i in range(self.N):
                for j in range(self.K):
                    log_writer.add_histogram(
                        prefix + "/latent/q_z_xy_logit{0}{1}".format(i, j),
                        self.q_dist.logits[:, i, j],
                        curr_iter,
                    )


class AdditiveAttention(nn.Module):
    # Implementing the attention module of Bahdanau et al. 2015 where
    # score(h_j, s_(i-1)) = v . tanh(W_1 h_j + W_2 s_(i-1))
    def __init__(
        self, encoder_hidden_state_dim, decoder_hidden_state_dim, internal_dim=None
    ):
        super(AdditiveAttention, self).__init__()

        if internal_dim is None:
            internal_dim = int(
                (encoder_hidden_state_dim + decoder_hidden_state_dim) / 2
            )

        self.w1 = nn.Linear(encoder_hidden_state_dim, internal_dim, bias=False)
        self.w2 = nn.Linear(decoder_hidden_state_dim, internal_dim, bias=False)
        self.v = nn.Linear(internal_dim, 1, bias=False)

    def score(self, encoder_state, decoder_state):
        # encoder_state is of shape (batch, enc_dim)
        # decoder_state is of shape (batch, dec_dim)
        # return value should be of shape (batch, 1)
        return self.v(torch.tanh(self.w1(encoder_state) + self.w2(decoder_state)))

    def forward(self, encoder_states, decoder_state):
        # encoder_states is of shape (batch, num_enc_states, enc_dim)
        # decoder_state is of shape (batch, dec_dim)
        score_vec = torch.cat(
            [
                self.score(encoder_states[:, i], decoder_state)
                for i in range(encoder_states.shape[1])
            ],
            dim=1,
        )
        # score_vec is of shape (batch, num_enc_states)

        attention_probs = torch.unsqueeze(F.softmax(score_vec, dim=1), dim=2)
        # attention_probs is of shape (batch, num_enc_states, 1)

        final_context_vec = torch.sum(attention_probs * encoder_states, dim=1)
        # final_context_vec is of shape (batch, enc_dim)

        return final_context_vec, attention_probs


class TemporallyBatchedAdditiveAttention(AdditiveAttention):
    # Implementing the attention module of Bahdanau et al. 2015 where
    # score(h_j, s_(i-1)) = v . tanh(W_1 h_j + W_2 s_(i-1))
    def __init__(
        self, encoder_hidden_state_dim, decoder_hidden_state_dim, internal_dim=None
    ):
        super(TemporallyBatchedAdditiveAttention, self).__init__(
            encoder_hidden_state_dim, decoder_hidden_state_dim, internal_dim
        )

    def score(self, encoder_state, decoder_state):
        # encoder_state is of shape (batch, num_enc_states, max_time, enc_dim)
        # decoder_state is of shape (batch, max_time, dec_dim)
        # return value should be of shape (batch, num_enc_states, max_time, 1)
        return self.v(
            torch.tanh(
                self.w1(encoder_state) + torch.unsqueeze(self.w2(decoder_state), dim=1)
            )
        )

    def forward(self, encoder_states, decoder_state):
        # encoder_states is of shape (batch, num_enc_states, max_time, enc_dim)
        # decoder_state is of shape (batch, max_time, dec_dim)
        score_vec = self.score(encoder_states, decoder_state)
        # score_vec is of shape (batch, num_enc_states, max_time, 1)

        attention_probs = F.softmax(score_vec, dim=1)
        # attention_probs is of shape (batch, num_enc_states, max_time, 1)

        final_context_vec = torch.sum(attention_probs * encoder_states, dim=1)
        # final_context_vec is of shape (batch, max_time, enc_dim)

        return final_context_vec, torch.squeeze(
            torch.transpose(attention_probs, 1, 2), dim=3
        )


class CNNMapEncoder(nn.Module):
    def __init__(
        self, map_channels, hidden_channels, output_size, masks, strides, patch_size
    ):
        super(CNNMapEncoder, self).__init__()
        self.convs = nn.ModuleList()
        patch_size_x = patch_size[0] + patch_size[2]
        patch_size_y = patch_size[1] + patch_size[3]
        input_size = (map_channels, patch_size_x, patch_size_y)
        x_dummy = torch.ones(input_size).unsqueeze(0) * torch.tensor(float("nan"))

        for i, hidden_size in enumerate(hidden_channels):
            self.convs.append(
                nn.Conv2d(
                    map_channels if i == 0 else hidden_channels[i - 1],
                    hidden_channels[i],
                    masks[i],
                    stride=strides[i],
                )
            )
            x_dummy = self.convs[i](x_dummy)

        self.fc = nn.Linear(x_dummy.numel(), output_size)

    def forward(self, x, training):
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3, decay=1.):
    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize) * decay**it

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x))

    return lr_lambda


def to_one_hot(labels, n_labels):
    return torch.eye(n_labels, device=labels.device)[labels]


def exp_anneal(anneal_kws):
    device = anneal_kws['device']
    start = torch.tensor(anneal_kws['start'], device=device)
    finish = torch.tensor(anneal_kws['finish'], device=device)
    rate = torch.tensor(anneal_kws['rate'], device=device)
    return lambda step: finish - (finish - start)*torch.pow(rate, torch.tensor(step, dtype=torch.float, device=device))


def sigmoid_anneal(anneal_kws):
    device = anneal_kws['device']
    start = torch.tensor(anneal_kws['start'], device=device)
    finish = torch.tensor(anneal_kws['finish'], device=device)
    center_step = torch.tensor(anneal_kws['center_step'], device=device, dtype=torch.float)
    steps_lo_to_hi = torch.tensor(anneal_kws['steps_lo_to_hi'], device=device, dtype=torch.float)
    return lambda step: start + (finish - start)*torch.sigmoid((torch.tensor(float(step), device=device) - center_step) * (1./steps_lo_to_hi))


class CustomLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        super(CustomLR, self).__init__(optimizer, lr_lambda, last_epoch)

    def get_lr(self):
        return [lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]


def mutual_inf_mc(x_dist):
    dist = x_dist.__class__
    H_y = dist(probs=x_dist.probs.mean(dim=0)).entropy()
    return (H_y - x_dist.entropy().mean(dim=0)).sum()


def run_lstm_on_variable_length_seqs(lstm_module, original_seqs, lower_indices=None, upper_indices=None, total_length=None):
    bs, tf = original_seqs.shape[:2]
    if lower_indices is None:
        lower_indices = torch.zeros(bs, dtype=torch.int)
    if upper_indices is None:
        upper_indices = torch.ones(bs, dtype=torch.int) * (tf - 1)
    if total_length is None:
        total_length = max(upper_indices) + 1
    # This is done so that we can just pass in self.prediction_timesteps
    # (which we want to INCLUDE, so this will exclude the next timestep).
    inclusive_break_indices = upper_indices + 1

    pad_list = list()
    for i, seq_len in enumerate(inclusive_break_indices):
        pad_list.append(original_seqs[i, lower_indices[i]:seq_len])

    packed_seqs = rnn.pack_sequence(pad_list, enforce_sorted=False)
    packed_output, (h_n, c_n) = lstm_module(packed_seqs)
    output, _ = rnn.pad_packed_sequence(packed_output,
                                        batch_first=True,
                                        total_length=total_length)

    return output, (h_n, c_n)


def extract_subtensor_per_batch_element(tensor, indices):
    batch_idxs = torch.arange(start=0, end=len(indices))

    batch_idxs = batch_idxs[~torch.isnan(indices)]
    indices = indices[~torch.isnan(indices)]
    if indices.size == 0:
        return None
    else:
        indices = indices.long()
    if tensor.is_cuda:
        batch_idxs = batch_idxs.to(tensor.get_device())
        indices = indices.to(tensor.get_device())
    return tensor[batch_idxs, indices]


def unpack_RNN_state(state_tuple):
    # PyTorch returned LSTM states have 3 dims:
    # (num_layers * num_directions, batch, hidden_size)

    state = torch.cat(state_tuple, dim=0).permute(1, 0, 2)
    # Now state is (batch, 2 * num_layers * num_directions, hidden_size)

    state_size = state.size()
    return torch.reshape(state, (-1, state_size[1] * state_size[2]))

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# using wonder's beautiful simplification:
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))
class Dynamic(object):
    def __init__(self, dt, dyn_limits, device, model_registrar, xz_size, node_type):
        self.dt = dt
        self.device = device
        self.dyn_limits = dyn_limits
        self.initial_conditions = None
        self.model_registrar = model_registrar
        self.node_type = node_type
        self.init_constants()
        self.create_graph(xz_size)

    def set_initial_condition(self, init_con):
        self.initial_conditions = init_con

    def init_constants(self):
        pass

    def create_graph(self, xz_size):
        pass

    def integrate_samples(self, s, x):
        raise NotImplementedError

    def integrate_distribution(self, dist, x):
        raise NotImplementedError

    def create_graph(self, xz_size):
        pass
class MultimodalGenerativeCVAE(object):
    def __init__(
        self,
        env,
        node_type,
        model_registrar,
        hyperparams,
        device,
        edge_types,
        log_writer=None,
    ):
        self.hyperparams = hyperparams
        self.env = env
        self.node_type = node_type
        self.model_registrar = model_registrar
        self.log_writer = log_writer
        self.device = device
        self.edge_types = [
            edge_type for edge_type in edge_types if edge_type[0] is node_type
        ]
        self.curr_iter = 0

        self.node_modules = dict()

        self.min_hl = self.hyperparams["minimum_history_length"]
        self.max_hl = self.hyperparams["maximum_history_length"]
        self.ph = self.hyperparams["prediction_horizon"]
        self.state = self.hyperparams["state"]
        self.pred_state = self.hyperparams["pred_state"][node_type]
        self.state_length = int(
            np.sum([len(entity_dims) for entity_dims in self.state[node_type].values()])
        )
        if self.hyperparams["incl_robot_node"]:
            self.robot_state_length = int(
                np.sum(
                    [
                        len(entity_dims)
                        for entity_dims in self.state[env.robot_type].values()
                    ]
                )
            )
        self.pred_state_length = int(
            np.sum([len(entity_dims) for entity_dims in self.pred_state.values()])
        )

        edge_types_str = [
            DirectedEdge.get_str_from_types(*edge_type) for edge_type in self.edge_types
        ]
        self.create_graphical_model(edge_types_str)

        dynamic_class = getattr(Dynamic, hyperparams["dynamic"][self.node_type]["name"])
        dyn_limits = hyperparams["dynamic"][self.node_type]["limits"]
        self.dynamic = dynamic_class(
            self.env.scenes[0].dt,
            dyn_limits,
            device,
            self.model_registrar,
            self.x_size,
            self.node_type,
        )

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter

    def add_submodule(self, name, model_if_absent):
        self.node_modules[name] = self.model_registrar.get_model(name, model_if_absent)

    def clear_submodules(self):
        self.node_modules.clear()

    def create_node_models(self):
        ############################
        #   Node History Encoder   #
        ############################
        self.add_submodule(
            self.node_type + "/node_history_encoder",
            model_if_absent=nn.LSTM(
                input_size=self.state_length,
                hidden_size=self.hyperparams["enc_rnn_dim_history"],
                batch_first=True,
            ),
        )

        ###########################
        #   Node Future Encoder   #
        ###########################
        # We'll create this here, but then later check if in training mode.
        # Based on that, we'll factor this into the computation graph (or not).
        self.add_submodule(
            self.node_type + "/node_future_encoder",
            model_if_absent=nn.LSTM(
                input_size=self.pred_state_length,
                hidden_size=self.hyperparams["enc_rnn_dim_future"],
                bidirectional=True,
                batch_first=True,
            ),
        )
        # These are related to how you initialize states for the node future encoder.
        self.add_submodule(
            self.node_type + "/node_future_encoder/initial_h",
            model_if_absent=nn.Linear(
                self.state_length, self.hyperparams["enc_rnn_dim_future"]
            ),
        )
        self.add_submodule(
            self.node_type + "/node_future_encoder/initial_c",
            model_if_absent=nn.Linear(
                self.state_length, self.hyperparams["enc_rnn_dim_future"]
            ),
        )

        ############################
        #   Robot Future Encoder   #
        ############################
        # We'll create this here, but then later check if we're next to the robot.
        # Based on that, we'll factor this into the computation graph (or not).
        if self.hyperparams["incl_robot_node"]:
            self.add_submodule(
                "robot_future_encoder",
                model_if_absent=nn.LSTM(
                    input_size=self.robot_state_length,
                    hidden_size=self.hyperparams["enc_rnn_dim_future"],
                    bidirectional=True,
                    batch_first=True,
                ),
            )
            # These are related to how you initialize states for the robot future encoder.
            self.add_submodule(
                "robot_future_encoder/initial_h",
                model_if_absent=nn.Linear(
                    self.robot_state_length, self.hyperparams["enc_rnn_dim_future"]
                ),
            )
            self.add_submodule(
                "robot_future_encoder/initial_c",
                model_if_absent=nn.Linear(
                    self.robot_state_length, self.hyperparams["enc_rnn_dim_future"]
                ),
            )

        if self.hyperparams["edge_encoding"]:
            ##############################
            #   Edge Influence Encoder   #
            ##############################
            # NOTE: The edge influence encoding happens during calls
            # to forward or incremental_forward, so we don't create
            # a model for it here for the max and sum variants.
            if self.hyperparams["edge_influence_combine_method"] == "bi-rnn":
                self.add_submodule(
                    self.node_type + "/edge_influence_encoder",
                    model_if_absent=nn.LSTM(
                        input_size=self.hyperparams["enc_rnn_dim_edge"],
                        hidden_size=self.hyperparams["enc_rnn_dim_edge_influence"],
                        bidirectional=True,
                        batch_first=True,
                    ),
                )

                # Four times because we're trying to mimic a bi-directional
                # LSTM's output (which, here, is c and h from both ends).
                self.eie_output_dims = (
                    4 * self.hyperparams["enc_rnn_dim_edge_influence"]
                )

            elif self.hyperparams["edge_influence_combine_method"] == "attention":
                # Chose additive attention because of https://arxiv.org/pdf/1703.03906.pdf
                # We calculate an attention context vector using the encoded edges as the "encoder"
                # (that we attend _over_)
                # and the node history encoder representation as the "decoder state" (that we attend _on_).
                self.add_submodule(
                    self.node_type + "/edge_influence_encoder",
                    model_if_absent=AdditiveAttention(
                        encoder_hidden_state_dim=self.hyperparams[
                            "enc_rnn_dim_edge_influence"
                        ],
                        decoder_hidden_state_dim=self.hyperparams[
                            "enc_rnn_dim_history"
                        ],
                    ),
                )

                self.eie_output_dims = self.hyperparams["enc_rnn_dim_edge_influence"]

        ###################
        #   Map Encoder   #
        ###################
        if self.hyperparams["use_map_encoding"]:
            if self.node_type in self.hyperparams["map_encoder"]:
                me_params = self.hyperparams["map_encoder"][self.node_type]
                self.add_submodule(
                    self.node_type + "/map_encoder",
                    model_if_absent=CNNMapEncoder(
                        me_params["map_channels"],
                        me_params["hidden_channels"],
                        me_params["output_size"],
                        me_params["masks"],
                        me_params["strides"],
                        me_params["patch_size"],
                    ),
                )

        ################################
        #   Discrete Latent Variable   #
        ################################
        self.latent = DiscreteLatent(self.hyperparams, self.device)

        ######################################################################
        #   Various Fully-Connected Layers from Encoder to Latent Variable   #
        ######################################################################
        # Node History Encoder
        x_size = self.hyperparams["enc_rnn_dim_history"]
        if self.hyperparams["edge_encoding"]:
            #              Edge Encoder
            x_size += self.eie_output_dims
        if self.hyperparams["incl_robot_node"]:
            #              Future Conditional Encoder
            x_size += 4 * self.hyperparams["enc_rnn_dim_future"]
        if (
            self.hyperparams["use_map_encoding"]
            and self.node_type in self.hyperparams["map_encoder"]
        ):
            #              Map Encoder
            x_size += self.hyperparams["map_encoder"][self.node_type]["output_size"]

        z_size = self.hyperparams["N"] * self.hyperparams["K"]

        if self.hyperparams["p_z_x_MLP_dims"] is not None:
            self.add_submodule(
                self.node_type + "/p_z_x",
                model_if_absent=nn.Linear(x_size, self.hyperparams["p_z_x_MLP_dims"]),
            )
            hx_size = self.hyperparams["p_z_x_MLP_dims"]
        else:
            hx_size = x_size

        self.add_submodule(
            self.node_type + "/hx_to_z",
            model_if_absent=nn.Linear(hx_size, self.latent.z_dim),
        )

        if self.hyperparams["q_z_xy_MLP_dims"] is not None:
            self.add_submodule(
                self.node_type + "/q_z_xy",
                #                                           Node Future Encoder
                model_if_absent=nn.Linear(
                    x_size + 4 * self.hyperparams["enc_rnn_dim_future"],
                    self.hyperparams["q_z_xy_MLP_dims"],
                ),
            )
            hxy_size = self.hyperparams["q_z_xy_MLP_dims"]
        else:
            #                           Node Future Encoder
            hxy_size = x_size + 4 * self.hyperparams["enc_rnn_dim_future"]

        self.add_submodule(
            self.node_type + "/hxy_to_z",
            model_if_absent=nn.Linear(hxy_size, self.latent.z_dim),
        )

        ####################
        #   Decoder LSTM   #
        ####################
        if self.hyperparams["incl_robot_node"]:
            decoder_input_dims = (
                self.pred_state_length + self.robot_state_length + z_size + x_size
            )
        else:
            decoder_input_dims = self.pred_state_length + z_size + x_size

        self.add_submodule(
            self.node_type + "/decoder/state_action",
            model_if_absent=nn.Sequential(
                nn.Linear(self.state_length, self.pred_state_length)
            ),
        )

        self.add_submodule(
            self.node_type + "/decoder/rnn_cell",
            model_if_absent=nn.GRUCell(
                decoder_input_dims, self.hyperparams["dec_rnn_dim"]
            ),
        )
        self.add_submodule(
            self.node_type + "/decoder/initial_h",
            model_if_absent=nn.Linear(z_size + x_size, self.hyperparams["dec_rnn_dim"]),
        )

        ###################
        #   Decoder GMM   #
        ###################
        self.add_submodule(
            self.node_type + "/decoder/proj_to_GMM_log_pis",
            model_if_absent=nn.Linear(
                self.hyperparams["dec_rnn_dim"], self.hyperparams["GMM_components"]
            ),
        )
        self.add_submodule(
            self.node_type + "/decoder/proj_to_GMM_mus",
            model_if_absent=nn.Linear(
                self.hyperparams["dec_rnn_dim"],
                self.hyperparams["GMM_components"] * self.pred_state_length,
            ),
        )
        self.add_submodule(
            self.node_type + "/decoder/proj_to_GMM_log_sigmas",
            model_if_absent=nn.Linear(
                self.hyperparams["dec_rnn_dim"],
                self.hyperparams["GMM_components"] * self.pred_state_length,
            ),
        )
        self.add_submodule(
            self.node_type + "/decoder/proj_to_GMM_corrs",
            model_if_absent=nn.Linear(
                self.hyperparams["dec_rnn_dim"], self.hyperparams["GMM_components"]
            ),
        )

        self.x_size = x_size
        self.z_size = z_size

    def create_edge_models(self, edge_types):
        for edge_type in edge_types:
            neighbor_state_length = int(
                np.sum(
                    [
                        len(entity_dims)
                        for entity_dims in self.state[edge_type.split("->")[1]].values()
                    ]
                )
            )
            if self.hyperparams["edge_state_combine_method"] == "pointnet":
                self.add_submodule(
                    edge_type + "/pointnet_encoder",
                    model_if_absent=nn.Sequential(
                        nn.Linear(self.state_length, 2 * self.state_length),
                        nn.ReLU(),
                        nn.Linear(2 * self.state_length, 2 * self.state_length),
                        nn.ReLU(),
                    ),
                )

                edge_encoder_input_size = 2 * self.state_length + self.state_length

            elif self.hyperparams["edge_state_combine_method"] == "attention":
                self.add_submodule(
                    self.node_type + "/edge_attention_combine",
                    model_if_absent=TemporallyBatchedAdditiveAttention(
                        encoder_hidden_state_dim=self.state_length,
                        decoder_hidden_state_dim=self.state_length,
                    ),
                )
                edge_encoder_input_size = self.state_length + neighbor_state_length

            else:
                edge_encoder_input_size = self.state_length + neighbor_state_length

            self.add_submodule(
                edge_type + "/edge_encoder",
                model_if_absent=nn.LSTM(
                    input_size=edge_encoder_input_size,
                    hidden_size=self.hyperparams["enc_rnn_dim_edge"],
                    batch_first=True,
                ),
            )

    def create_graphical_model(self, edge_types):
        """
        Creates or queries all trainable components.

        :param edge_types: List containing strings for all possible edge types for the node type.
        :return: None
        """
        self.clear_submodules()

        ############################
        #   Everything but Edges   #
        ############################
        self.create_node_models()

        #####################
        #   Edge Encoders   #
        #####################
        if self.hyperparams["edge_encoding"]:
            self.create_edge_models(edge_types)

        for name, module in self.node_modules.items():
            module.to(self.device)

    def create_new_scheduler(
        self, name, annealer, annealer_kws, creation_condition=True
    ):
        value_scheduler = None
        rsetattr(self, name + "_scheduler", value_scheduler)
        if creation_condition:
            annealer_kws["device"] = self.device
            value_annealer = annealer(annealer_kws)
            rsetattr(self, name + "_annealer", value_annealer)

            # This is the value that we'll update on each call of
            # step_annealers().
            rsetattr(self, name, value_annealer(0).clone().detach())
            dummy_optimizer = optim.Optimizer(
                [rgetattr(self, name)], {"lr": value_annealer(0).clone().detach()}
            )
            rsetattr(self, name + "_optimizer", dummy_optimizer)

            value_scheduler = CustomLR(dummy_optimizer, value_annealer)
            rsetattr(self, name + "_scheduler", value_scheduler)

        self.schedulers.append(value_scheduler)
        self.annealed_vars.append(name)

    def set_annealing_params(self):
        self.schedulers = list()
        self.annealed_vars = list()

        self.create_new_scheduler(
            name="kl_weight",
            annealer=sigmoid_anneal,
            annealer_kws={
                "start": self.hyperparams["kl_weight_start"],
                "finish": self.hyperparams["kl_weight"],
                "center_step": self.hyperparams["kl_crossover"],
                "steps_lo_to_hi": self.hyperparams["kl_crossover"]
                / self.hyperparams["kl_sigmoid_divisor"],
            },
        )

        self.create_new_scheduler(
            name="latent.temp",
            annealer=exp_anneal,
            annealer_kws={
                "start": self.hyperparams["tau_init"],
                "finish": self.hyperparams["tau_final"],
                "rate": self.hyperparams["tau_decay_rate"],
            },
        )

        self.create_new_scheduler(
            name="latent.z_logit_clip",
            annealer=sigmoid_anneal,
            annealer_kws={
                "start": self.hyperparams["z_logit_clip_start"],
                "finish": self.hyperparams["z_logit_clip_final"],
                "center_step": self.hyperparams["z_logit_clip_crossover"],
                "steps_lo_to_hi": self.hyperparams["z_logit_clip_crossover"]
                / self.hyperparams["z_logit_clip_divisor"],
            },
            creation_condition=self.hyperparams["use_z_logit_clipping"],
        )

    def step_annealers(self):
        # This should manage all of the step-wise changed
        # parameters automatically.
        for idx, annealed_var in enumerate(self.annealed_vars):
            if rgetattr(self, annealed_var + "_scheduler") is not None:
                # First we step the scheduler.
                with warnings.catch_warnings():  # We use a dummy optimizer: Warning because no .step() was called on it
                    warnings.simplefilter("ignore")
                    rgetattr(self, annealed_var + "_scheduler").step()

                # Then we set the annealed vars' value.
                rsetattr(
                    self,
                    annealed_var,
                    rgetattr(self, annealed_var + "_optimizer").param_groups[0]["lr"],
                )

        self.summarize_annealers()

    def summarize_annealers(self):
        if self.log_writer is not None:
            for annealed_var in self.annealed_vars:
                if rgetattr(self, annealed_var) is not None:
                    self.log_writer.add_scalar(
                        "%s/%s" % (str(self.node_type), annealed_var.replace(".", "/")),
                        rgetattr(self, annealed_var),
                        self.curr_iter,
                    )

    def obtain_encoded_tensors(
        self,
        mode,
        inputs,
        inputs_st,
        labels,
        labels_st,
        first_history_indices,
        neighbors,
        neighbors_edge_value,
        robot,
        map,
    ) -> (
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ):
        """
        Encodes input and output tensors for node and robot.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :return: tuple(x, x_nr_t, y_e, y_r, y, n_s_t0)
            WHERE
            - x: Encoded input / condition tensor to the CVAE x_e.
            - x_r_t: Robot state (if robot is in scene).
            - y_e: Encoded label / future of the node.
            - y_r: Encoded future of the robot.
            - y: Label / future of the node.
            - n_s_t0: Standardized current state of the node.
        """

        x, x_r_t, y_e, y_r, y = None, None, None, None, None
        initial_dynamics = dict()

        batch_size = inputs.shape[0]

        #########################################
        # Provide basic information to encoders #
        #########################################
        node_history = inputs
        node_present_state = inputs[:, -1]
        node_pos = inputs[:, -1, 0:2]
        node_vel = inputs[:, -1, 2:4]

        node_history_st = inputs_st
        node_present_state_st = inputs_st[:, -1]
        node_pos_st = inputs_st[:, -1, 0:2]
        node_vel_st = inputs_st[:, -1, 2:4]

        n_s_t0 = node_present_state_st

        initial_dynamics["pos"] = node_pos
        initial_dynamics["vel"] = node_vel

        self.dynamic.set_initial_condition(initial_dynamics)

        if self.hyperparams["incl_robot_node"]:
            x_r_t, y_r = robot[..., 0, :], robot[..., 1:, :]

        ##################
        # Encode History #
        ##################
        node_history_encoded = self.encode_node_history(
            mode, node_history_st, first_history_indices
        )

        ##################
        # Encode Present #
        ##################
        node_present = node_present_state_st  # [bs, state_dim]

        ##################
        # Encode Future #
        ##################
        if mode != ModeKeys.PREDICT:
            y = labels_st

        ##############################
        # Encode Node Edges per Type #
        ##############################
        if self.hyperparams["edge_encoding"]:
            node_edges_encoded = list()
            for edge_type in self.edge_types:
                # Encode edges for given edge type
                encoded_edges_type = self.encode_edge(
                    mode,
                    node_history,
                    node_history_st,
                    edge_type,
                    neighbors[edge_type],
                    neighbors_edge_value[edge_type],
                    first_history_indices,
                )
                node_edges_encoded.append(
                    encoded_edges_type
                )  # List of [bs/nbs, enc_rnn_dim]
            #####################
            # Encode Node Edges #
            #####################
            total_edge_influence = self.encode_total_edge_influence(
                mode, node_edges_encoded, node_history_encoded, batch_size
            )

        ################
        # Map Encoding #
        ################
        if (
            self.hyperparams["use_map_encoding"]
            and self.node_type in self.hyperparams["map_encoder"]
        ):
            if self.log_writer and (self.curr_iter + 1) % 500 == 0:
                map_clone = map.clone()
                map_patch = self.hyperparams["map_encoder"][self.node_type][
                    "patch_size"
                ]
                map_clone[
                    :,
                    :,
                    map_patch[1] - 5 : map_patch[1] + 5,
                    map_patch[0] - 5 : map_patch[0] + 5,
                ] = 1.0
                self.log_writer.add_images(
                    f"{self.node_type}/cropped_maps",
                    map_clone,
                    self.curr_iter,
                    dataformats="NCWH",
                )

            encoded_map = self.node_modules[self.node_type + "/map_encoder"](
                map * 2.0 - 1.0, (mode == ModeKeys.TRAIN)
            )
            do = self.hyperparams["map_encoder"][self.node_type]["dropout"]
            encoded_map = F.dropout(encoded_map, do, training=(mode == ModeKeys.TRAIN))

        ######################################
        # Concatenate Encoder Outputs into x #
        ######################################
        x_concat_list = list()

        # Every node has an edge-influence encoder (which could just be zero).
        if self.hyperparams["edge_encoding"]:
            x_concat_list.append(total_edge_influence)  # [bs/nbs, 4*enc_rnn_dim]

        # Every node has a history encoder.
        x_concat_list.append(node_history_encoded)  # [bs/nbs, enc_rnn_dim_history]

        if self.hyperparams["incl_robot_node"]:
            robot_future_encoder = self.encode_robot_future(mode, x_r_t, y_r)
            x_concat_list.append(robot_future_encoder)

        if (
            self.hyperparams["use_map_encoding"]
            and self.node_type in self.hyperparams["map_encoder"]
        ):
            if self.log_writer:
                self.log_writer.add_scalar(
                    f"{self.node_type}/encoded_map_max",
                    torch.max(torch.abs(encoded_map)),
                    self.curr_iter,
                )
            x_concat_list.append(encoded_map)

        x = torch.cat(x_concat_list, dim=1)

        if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
            y_e = self.encode_node_future(mode, node_present, y)

        return x, x_r_t, y_e, y_r, y, n_s_t0

    def encode_node_history(self, mode, node_hist, first_history_indices):
        """
        Encodes the nodes history.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_hist: Historic and current state of the node. [bs, mhl, state]
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :return: Encoded node history tensor. [bs, enc_rnn_dim]
        """
        outputs, _ = run_lstm_on_variable_length_seqs(
            self.node_modules[self.node_type + "/node_history_encoder"],
            original_seqs=node_hist,
            lower_indices=first_history_indices,
        )

        outputs = F.dropout(
            outputs,
            p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
            training=(mode == ModeKeys.TRAIN),
        )  # [bs, max_time, enc_rnn_dim]

        last_index_per_sequence = -(first_history_indices + 1)

        return outputs[
            torch.arange(first_history_indices.shape[0]), last_index_per_sequence
        ]

    def encode_edge(
        self,
        mode,
        node_history,
        node_history_st,
        edge_type,
        neighbors,
        neighbors_edge_value,
        first_history_indices,
    ):
        max_hl = self.hyperparams["maximum_history_length"]

        edge_states_list = list()  # list of [#of neighbors, max_ht, state_dim]
        for i, neighbor_states in enumerate(
            neighbors
        ):  # Get neighbors for timestep in batch
            if (
                len(neighbor_states) == 0
            ):  # There are no neighbors for edge type # TODO necessary?
                neighbor_state_length = int(
                    np.sum(
                        [
                            len(entity_dims)
                            for entity_dims in self.state[edge_type[1]].values()
                        ]
                    )
                )
                edge_states_list.append(
                    torch.zeros(
                        (1, max_hl + 1, neighbor_state_length), device=self.device
                    )
                )
            else:
                edge_states_list.append(
                    torch.stack(neighbor_states, dim=0).to(self.device)
                )

        if self.hyperparams["edge_state_combine_method"] == "sum":
            # Used in Structural-RNN to combine edges as well.
            op_applied_edge_states_list = list()
            for neighbors_state in edge_states_list:
                op_applied_edge_states_list.append(torch.sum(neighbors_state, dim=0))
            combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)
            if self.hyperparams["dynamic_edges"] == "yes":
                # Should now be (bs, time, 1)
                op_applied_edge_mask_list = list()
                for edge_value in neighbors_edge_value:
                    op_applied_edge_mask_list.append(
                        torch.clamp(
                            torch.sum(edge_value.to(self.device), dim=0, keepdim=True),
                            max=1.0,
                        )
                    )
                combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0)

        elif self.hyperparams["edge_state_combine_method"] == "max":
            # Used in NLP, e.g. max over word embeddings in a sentence.
            op_applied_edge_states_list = list()
            for neighbors_state in edge_states_list:
                op_applied_edge_states_list.append(torch.max(neighbors_state, dim=0))
            combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)
            if self.hyperparams["dynamic_edges"] == "yes":
                # Should now be (bs, time, 1)
                op_applied_edge_mask_list = list()
                for edge_value in neighbors_edge_value:
                    op_applied_edge_mask_list.append(
                        torch.clamp(
                            torch.max(edge_value.to(self.device), dim=0, keepdim=True),
                            max=1.0,
                        )
                    )
                combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0)

        elif self.hyperparams["edge_state_combine_method"] == "mean":
            # Used in NLP, e.g. mean over word embeddings in a sentence.
            op_applied_edge_states_list = list()
            for neighbors_state in edge_states_list:
                op_applied_edge_states_list.append(torch.mean(neighbors_state, dim=0))
            combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)
            if self.hyperparams["dynamic_edges"] == "yes":
                # Should now be (bs, time, 1)
                op_applied_edge_mask_list = list()
                for edge_value in neighbors_edge_value:
                    op_applied_edge_mask_list.append(
                        torch.clamp(
                            torch.mean(edge_value.to(self.device), dim=0, keepdim=True),
                            max=1.0,
                        )
                    )
                combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0)

        joint_history = torch.cat([combined_neighbors, node_history_st], dim=-1)

        outputs, _ = run_lstm_on_variable_length_seqs(
            self.node_modules[
                DirectedEdge.get_str_from_types(*edge_type) + "/edge_encoder"
            ],
            original_seqs=joint_history,
            lower_indices=first_history_indices,
        )

        outputs = F.dropout(
            outputs,
            p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
            training=(mode == ModeKeys.TRAIN),
        )  # [bs, max_time, enc_rnn_dim]

        last_index_per_sequence = -(first_history_indices + 1)
        ret = outputs[
            torch.arange(last_index_per_sequence.shape[0]), last_index_per_sequence
        ]
        if self.hyperparams["dynamic_edges"] == "yes":
            return ret * combined_edge_masks
        else:
            return ret

    def encode_total_edge_influence(
        self, mode, encoded_edges, node_history_encoder, batch_size
    ):
        if self.hyperparams["edge_influence_combine_method"] == "sum":
            stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
            combined_edges = torch.sum(stacked_encoded_edges, dim=0)

        elif self.hyperparams["edge_influence_combine_method"] == "mean":
            stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
            combined_edges = torch.mean(stacked_encoded_edges, dim=0)

        elif self.hyperparams["edge_influence_combine_method"] == "max":
            stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
            combined_edges = torch.max(stacked_encoded_edges, dim=0)

        elif self.hyperparams["edge_influence_combine_method"] == "bi-rnn":
            if len(encoded_edges) == 0:
                combined_edges = torch.zeros(
                    (batch_size, self.eie_output_dims), device=self.device
                )

            else:
                # axis=1 because then we get size [batch_size, max_time, depth]
                encoded_edges = torch.stack(encoded_edges, dim=1)

                _, state = self.node_modules[
                    self.node_type + "/edge_influence_encoder"
                ](encoded_edges)
                combined_edges = unpack_RNN_state(state)
                combined_edges = F.dropout(
                    combined_edges,
                    p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
                    training=(mode == ModeKeys.TRAIN),
                )

        elif self.hyperparams["edge_influence_combine_method"] == "attention":
            # Used in Social Attention (https://arxiv.org/abs/1710.04689)
            if len(encoded_edges) == 0:
                combined_edges = torch.zeros(
                    (batch_size, self.eie_output_dims), device=self.device
                )

            else:
                # axis=1 because then we get size [batch_size, max_time, depth]
                encoded_edges = torch.stack(encoded_edges, dim=1)
                combined_edges, _ = self.node_modules[
                    self.node_type + "/edge_influence_encoder"
                ](encoded_edges, node_history_encoder)
                combined_edges = F.dropout(
                    combined_edges,
                    p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
                    training=(mode == ModeKeys.TRAIN),
                )

        return combined_edges

    def encode_node_future(self, mode, node_present, node_future) -> torch.Tensor:
        """
        Encodes the node future (during training) using a bi-directional LSTM

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_present: Current state of the node. [bs, state]
        :param node_future: Future states of the node. [bs, ph, state]
        :return: Encoded future.
        """
        initial_h_model = self.node_modules[
            self.node_type + "/node_future_encoder/initial_h"
        ]
        initial_c_model = self.node_modules[
            self.node_type + "/node_future_encoder/initial_c"
        ]

        # Here we're initializing the forward hidden states,
        # but zeroing the backward ones.
        initial_h = initial_h_model(node_present)
        initial_h = torch.stack(
            [initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0
        )

        initial_c = initial_c_model(node_present)
        initial_c = torch.stack(
            [initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0
        )

        initial_state = (initial_h, initial_c)

        _, state = self.node_modules[self.node_type + "/node_future_encoder"](
            node_future, initial_state
        )
        state = unpack_RNN_state(state)
        state = F.dropout(
            state,
            p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
            training=(mode == ModeKeys.TRAIN),
        )

        return state

    def encode_robot_future(self, mode, robot_present, robot_future) -> torch.Tensor:
        """
        Encodes the robot future (during training) using a bi-directional LSTM

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param robot_present: Current state of the robot. [bs, state]
        :param robot_future: Future states of the robot. [bs, ph, state]
        :return: Encoded future.
        """
        initial_h_model = self.node_modules["robot_future_encoder/initial_h"]
        initial_c_model = self.node_modules["robot_future_encoder/initial_c"]

        # Here we're initializing the forward hidden states,
        # but zeroing the backward ones.
        initial_h = initial_h_model(robot_present)
        initial_h = torch.stack(
            [initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0
        )

        initial_c = initial_c_model(robot_present)
        initial_c = torch.stack(
            [initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0
        )

        initial_state = (initial_h, initial_c)

        _, state = self.node_modules["robot_future_encoder"](
            robot_future, initial_state
        )
        state = unpack_RNN_state(state)
        state = F.dropout(
            state,
            p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
            training=(mode == ModeKeys.TRAIN),
        )

        return state

    def q_z_xy(self, mode, x, y_e) -> torch.Tensor:
        r"""
        .. math:: q_\phi(z \mid \mathbf{x}_i, \mathbf{y}_i)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y_e: Encoded future tensor.
        :return: Latent distribution of the CVAE.
        """
        xy = torch.cat([x, y_e], dim=1)

        if self.hyperparams["q_z_xy_MLP_dims"] is not None:
            dense = self.node_modules[self.node_type + "/q_z_xy"]
            h = F.dropout(
                F.relu(dense(xy)),
                p=1.0 - self.hyperparams["MLP_dropout_keep_prob"],
                training=(mode == ModeKeys.TRAIN),
            )

        else:
            h = xy

        to_latent = self.node_modules[self.node_type + "/hxy_to_z"]
        return self.latent.dist_from_h(to_latent(h), mode)

    def p_z_x(self, mode, x):
        r"""
        .. math:: p_\theta(z \mid \mathbf{x}_i)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :return: Latent distribution of the CVAE.
        """
        if self.hyperparams["p_z_x_MLP_dims"] is not None:
            dense = self.node_modules[self.node_type + "/p_z_x"]
            h = F.dropout(
                F.relu(dense(x)),
                p=1.0 - self.hyperparams["MLP_dropout_keep_prob"],
                training=(mode == ModeKeys.TRAIN),
            )

        else:
            h = x

        to_latent = self.node_modules[self.node_type + "/hx_to_z"]
        return self.latent.dist_from_h(to_latent(h), mode)

    def project_to_GMM_params(
        self, tensor
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Projects tensor to parameters of a GMM with N components and D dimensions.

        :param tensor: Input tensor.
        :return: tuple(log_pis, mus, log_sigmas, corrs)
            WHERE
            - log_pis: Weight (logarithm) of each GMM component. [N]
            - mus: Mean of each GMM component. [N, D]
            - log_sigmas: Standard Deviation (logarithm) of each GMM component. [N, D]
            - corrs: Correlation between the GMM components. [N]
        """
        log_pis = self.node_modules[self.node_type + "/decoder/proj_to_GMM_log_pis"](
            tensor
        )
        mus = self.node_modules[self.node_type + "/decoder/proj_to_GMM_mus"](tensor)
        log_sigmas = self.node_modules[
            self.node_type + "/decoder/proj_to_GMM_log_sigmas"
        ](tensor)
        corrs = torch.tanh(
            self.node_modules[self.node_type + "/decoder/proj_to_GMM_corrs"](tensor)
        )
        return log_pis, mus, log_sigmas, corrs

    def p_y_xz(
        self,
        mode,
        x,
        x_nr_t,
        y_r,
        n_s_t0,
        z_stacked,
        prediction_horizon,
        num_samples,
        num_components=1,
        gmm_mode=False,
    ):
        r"""
        .. math:: p_\psi(\mathbf{y}_i \mid \mathbf{x}_i, z)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param x_nr_t: Joint state of node and robot (if robot is in scene).
        :param y: Future tensor.
        :param y_r: Encoded future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z_stacked: Stacked latent state. [num_samples_z * num_samples_gmm, bs, latent_state]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param num_components: Number of GMM components.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :return: GMM2D. If mode is Predict, also samples from the GMM.
        """
        ph = prediction_horizon
        pred_dim = self.pred_state_length

        z = torch.reshape(z_stacked, (-1, self.latent.z_dim))
        zx = torch.cat([z, x.repeat(num_samples * num_components, 1)], dim=1)

        cell = self.node_modules[self.node_type + "/decoder/rnn_cell"]
        initial_h_model = self.node_modules[self.node_type + "/decoder/initial_h"]

        initial_state = initial_h_model(zx)

        log_pis, mus, log_sigmas, corrs, a_sample = [], [], [], [], []

        # Infer initial action state for node from current state
        a_0 = self.node_modules[self.node_type + "/decoder/state_action"](n_s_t0)

        state = initial_state
        if self.hyperparams["incl_robot_node"]:
            input_ = torch.cat(
                [
                    zx,
                    a_0.repeat(num_samples * num_components, 1),
                    x_nr_t.repeat(num_samples * num_components, 1),
                ],
                dim=1,
            )
        else:
            input_ = torch.cat([zx, a_0.repeat(num_samples * num_components, 1)], dim=1)

        for j in range(ph):
            h_state = cell(input_, state)
            log_pi_t, mu_t, log_sigma_t, corr_t = self.project_to_GMM_params(h_state)

            gmm = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t)  # [k;bs, pred_dim]

            if mode == ModeKeys.PREDICT and gmm_mode:
                a_t = gmm.mode()
            else:
                a_t = gmm.rsample()

            if num_components > 1:
                if mode == ModeKeys.PREDICT:
                    log_pis.append(self.latent.p_dist.logits.repeat(num_samples, 1, 1))
                else:
                    log_pis.append(self.latent.q_dist.logits.repeat(num_samples, 1, 1))
            else:
                log_pis.append(
                    torch.ones_like(
                        corr_t.reshape(num_samples, num_components, -1)
                        .permute(0, 2, 1)
                        .reshape(-1, 1)
                    )
                )

            mus.append(
                mu_t.reshape(num_samples, num_components, -1, 2)
                .permute(0, 2, 1, 3)
                .reshape(-1, 2 * num_components)
            )
            log_sigmas.append(
                log_sigma_t.reshape(num_samples, num_components, -1, 2)
                .permute(0, 2, 1, 3)
                .reshape(-1, 2 * num_components)
            )
            corrs.append(
                corr_t.reshape(num_samples, num_components, -1)
                .permute(0, 2, 1)
                .reshape(-1, num_components)
            )

            if self.hyperparams["incl_robot_node"]:
                dec_inputs = [
                    zx,
                    a_t,
                    y_r[:, j].repeat(num_samples * num_components, 1),
                ]
            else:
                dec_inputs = [zx, a_t]
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state

        log_pis = torch.stack(log_pis, dim=1)
        mus = torch.stack(mus, dim=1)
        log_sigmas = torch.stack(log_sigmas, dim=1)
        corrs = torch.stack(corrs, dim=1)

        a_dist = GMM2D(
            torch.reshape(log_pis, [num_samples, -1, ph, num_components]),
            torch.reshape(mus, [num_samples, -1, ph, num_components * pred_dim]),
            torch.reshape(log_sigmas, [num_samples, -1, ph, num_components * pred_dim]),
            torch.reshape(corrs, [num_samples, -1, ph, num_components]),
        )

        if self.hyperparams["dynamic"][self.node_type]["distribution"]:
            y_dist = self.dynamic.integrate_distribution(a_dist, x)
        else:
            y_dist = a_dist

        if mode == ModeKeys.PREDICT:
            if gmm_mode:
                a_sample = a_dist.mode()
            else:
                a_sample = a_dist.rsample()
            sampled_future = self.dynamic.integrate_samples(a_sample, x)
            return y_dist, sampled_future
        else:
            return y_dist

    def encoder(self, mode, x, y_e, num_samples=None):
        """
        Encoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y_e: Encoded future tensor.
        :param num_samples: Number of samples from the latent space during Prediction.
        :return: tuple(z, kl_obj)
            WHERE
            - z: Samples from the latent space.
            - kl_obj: KL Divergenze between q and p
        """
        if mode == ModeKeys.TRAIN:
            sample_ct = self.hyperparams["k"]
        elif mode == ModeKeys.EVAL:
            sample_ct = self.hyperparams["k_eval"]
        elif mode == ModeKeys.PREDICT:
            sample_ct = num_samples
            if num_samples is None:
                raise ValueError("num_samples cannot be None with mode == PREDICT.")

        self.latent.q_dist = self.q_z_xy(mode, x, y_e)
        self.latent.p_dist = self.p_z_x(mode, x)

        z = self.latent.sample_q(sample_ct, mode)

        if mode == ModeKeys.TRAIN:
            kl_obj = self.latent.kl_q_p(
                self.log_writer, "%s" % str(self.node_type), self.curr_iter
            )
            if self.log_writer is not None:
                self.log_writer.add_scalar(
                    "%s/%s" % (str(self.node_type), "kl"), kl_obj, self.curr_iter
                )
        else:
            kl_obj = None

        return z, kl_obj

    def decoder(
        self,
        mode,
        x,
        x_nr_t,
        y,
        y_r,
        n_s_t0,
        z,
        labels,
        prediction_horizon,
        num_samples,
    ):
        """
        Decoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param x: Input / Condition tensor.
        :param x_nr_t: Joint state of node and robot (if robot is in scene).
        :param y: Future tensor.
        :param y_r: Encoded future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z: Stacked latent state.
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :return: Log probability of y over p.
        """

        num_components = self.hyperparams["N"] * self.hyperparams["K"]
        y_dist = self.p_y_xz(
            mode,
            x,
            x_nr_t,
            y_r,
            n_s_t0,
            z,
            prediction_horizon,
            num_samples,
            num_components=num_components,
        )
        log_p_yt_xz = torch.clamp(
            y_dist.log_prob(labels), max=self.hyperparams["log_p_yt_xz_max"]
        )
        if self.hyperparams["log_histograms"] and self.log_writer is not None:
            self.log_writer.add_histogram(
                "%s/%s" % (str(self.node_type), "log_p_yt_xz"),
                log_p_yt_xz,
                self.curr_iter,
            )

        log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)
        return log_p_y_xz

    def train_loss(
        self,
        inputs,
        inputs_st,
        first_history_indices,
        labels,
        labels_st,
        neighbors,
        neighbors_edge_value,
        robot,
        map,
        prediction_horizon,
    ) -> torch.Tensor:
        """
        Calculates the training loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :return: Scalar tensor -> nll loss
        """
        mode = ModeKeys.TRAIN

        x, x_nr_t, y_e, y_r, y, n_s_t0 = self.obtain_encoded_tensors(
            mode=mode,
            inputs=inputs,
            inputs_st=inputs_st,
            labels=labels,
            labels_st=labels_st,
            first_history_indices=first_history_indices,
            neighbors=neighbors,
            neighbors_edge_value=neighbors_edge_value,
            robot=robot,
            map=map,
        )

        z, kl = self.encoder(mode, x, y_e)
        log_p_y_xz = self.decoder(
            mode,
            x,
            x_nr_t,
            y,
            y_r,
            n_s_t0,
            z,
            labels,  # Loss is calculated on unstandardized label
            prediction_horizon,
            self.hyperparams["k"],
        )

        log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]
        log_likelihood = torch.mean(log_p_y_xz_mean)

        mutual_inf_q = mutual_inf_mc(self.latent.q_dist)
        mutual_inf_p = mutual_inf_mc(self.latent.p_dist)

        ELBO = log_likelihood - self.kl_weight * kl + 1.0 * mutual_inf_p
        loss = -ELBO

        if self.hyperparams["log_histograms"] and self.log_writer is not None:
            self.log_writer.add_histogram(
                "%s/%s" % (str(self.node_type), "log_p_y_xz"),
                log_p_y_xz_mean,
                self.curr_iter,
            )

        if self.log_writer is not None:
            self.log_writer.add_scalar(
                "%s/%s" % (str(self.node_type), "mutual_information_q"),
                mutual_inf_q,
                self.curr_iter,
            )
            self.log_writer.add_scalar(
                "%s/%s" % (str(self.node_type), "mutual_information_p"),
                mutual_inf_p,
                self.curr_iter,
            )
            self.log_writer.add_scalar(
                "%s/%s" % (str(self.node_type), "log_likelihood"),
                log_likelihood,
                self.curr_iter,
            )
            self.log_writer.add_scalar(
                "%s/%s" % (str(self.node_type), "loss"), loss, self.curr_iter
            )
            if self.hyperparams["log_histograms"]:
                self.latent.summarize_for_tensorboard(
                    self.log_writer, str(self.node_type), self.curr_iter
                )
        return loss

    def eval_loss(
        self,
        inputs,
        inputs_st,
        first_history_indices,
        labels,
        labels_st,
        neighbors,
        neighbors_edge_value,
        robot,
        map,
        prediction_horizon,
    ) -> torch.Tensor:
        """
        Calculates the evaluation loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :return: tuple(nll_q_is, nll_p, nll_exact, nll_sampled)
        """

        mode = ModeKeys.EVAL

        x, x_nr_t, y_e, y_r, y, n_s_t0 = self.obtain_encoded_tensors(
            mode=mode,
            inputs=inputs,
            inputs_st=inputs_st,
            labels=labels,
            labels_st=labels_st,
            first_history_indices=first_history_indices,
            neighbors=neighbors,
            neighbors_edge_value=neighbors_edge_value,
            robot=robot,
            map=map,
        )

        num_components = self.hyperparams["N"] * self.hyperparams["K"]
        ### Importance sampled NLL estimate
        z, _ = self.encoder(mode, x, y_e)  # [k_eval, nbs, N*K]
        z = self.latent.sample_p(1, mode, full_dist=True)
        y_dist, _ = self.p_y_xz(
            ModeKeys.PREDICT,
            x,
            x_nr_t,
            y_r,
            n_s_t0,
            z,
            prediction_horizon,
            num_samples=1,
            num_components=num_components,
        )
        # We use unstandardized labels to compute the loss
        log_p_yt_xz = torch.clamp(
            y_dist.log_prob(labels), max=self.hyperparams["log_p_yt_xz_max"]
        )
        log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)
        log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]
        log_likelihood = torch.mean(log_p_y_xz_mean)
        nll = -log_likelihood

        return nll

    def predict(
        self,
        inputs,
        inputs_st,
        first_history_indices,
        neighbors,
        neighbors_edge_value,
        robot,
        map,
        prediction_horizon,
        num_samples,
        z_mode=False,
        gmm_mode=False,
        full_dist=True,
        all_z_sep=False,
    ):
        """
        Predicts the future of a batch of nodes.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param z_mode: If True: Select the most likely latent state.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :param all_z_sep: Samples each latent mode individually without merging them into a GMM.
        :param full_dist: Samples all latent states and merges them into a GMM as output.
        :return:
        """
        mode = ModeKeys.PREDICT

        x, x_nr_t, _, y_r, _, n_s_t0 = self.obtain_encoded_tensors(
            mode=mode,
            inputs=inputs,
            inputs_st=inputs_st,
            labels=None,
            labels_st=None,
            first_history_indices=first_history_indices,
            neighbors=neighbors,
            neighbors_edge_value=neighbors_edge_value,
            robot=robot,
            map=map,
        )

        self.latent.p_dist = self.p_z_x(mode, x)
        z, num_samples, num_components = self.latent.sample_p(
            num_samples,
            mode,
            most_likely_z=z_mode,
            full_dist=full_dist,
            all_z_sep=all_z_sep,
        )

        _, our_sampled_future = self.p_y_xz(
            mode,
            x,
            x_nr_t,
            y_r,
            n_s_t0,
            z,
            prediction_horizon,
            num_samples,
            num_components,
            gmm_mode,
        )

        return our_sampled_future


#######################################################################################################################