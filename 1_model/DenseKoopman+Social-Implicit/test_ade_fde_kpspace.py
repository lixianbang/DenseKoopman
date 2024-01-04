import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import glob
from utils import *
from metrics import *
from model import SocialImplicit
from CFG import CFG


def test(KSTEPS=20):

    global loader_test, model, ROBUSTNESS
    model.eval()
    ade_bigls = []
    fde_bigls = []
    step = 0
    for batch in loader_test:
        step += 1
        #Get data
        batch = [tensor.cuda().double() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask,V_obs,A_obs,V_tr,A_tr = batch

        num_of_objs = obs_traj_rel.shape[1]

        V_tr = V_tr.squeeze()

        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(
            V_obs.data.cpu().numpy().squeeze(), V_x[0, :, :].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(
            V_tr.data.cpu().numpy().squeeze(), V_x[-1, :, :].copy())

        for n in range(num_of_objs):
            ade_ls[n] = []
            fde_ls[n] = []

        V_predx = model(V_obs_tmp, obs_traj, KSTEPS=KSTEPS)

        for k in range(KSTEPS):
            V_pred = V_predx[k:k + 1, ...]

            V_pred = V_pred.permute(0, 2, 3, 1)

            V_pred = V_pred.squeeze()

            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(
                V_pred.data.cpu().numpy().squeeze(), V_x[-1, :, :].copy())
            #Sensitivity
            V_pred_rel_to_abs += ROBUSTNESS

            for n in range(num_of_objs):
                pred = []
                target = []
                obsrvs = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:, n:n + 1, :])
                target.append(V_y_rel_to_abs[:, n:n + 1, :])
                obsrvs.append(V_x_rel_to_abs[:, n:n + 1, :])
                number_of.append(1)

                ade_ls[n].append(ade_kpspace(pred, target, number_of))
                fde_ls[n].append(fde_kpspace(pred, target, number_of))

        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)
    return ade_, fde_

if __name__ == '__main__':
    for ROBUSTNESS in [0]:  #[-0.1, -0.01, 0, +0.01, +0.1]:
        # print("*" * 30)
        # print("*" * 30)
        # print("ROBUSTNESS:", ROBUSTNESS)
        # print("*" * 30)
        # print("*" * 30)

        paths = [
            './checkpoint/Ab_Ex_only_zerocentered'

        ]
        KSTEPS = 20

        EASY_RESULTS = []

        # print("*" * 50)
        # print('Number of samples:', KSTEPS)
        # print("*" * 50)

        for feta in range(len(paths)):

            ade_ls = []
            fde_ls = []
            exp_ls = []
            path = paths[feta]
            exps = glob.glob(path)
            exps.sort()
            # print('Models being tested are:', exps)

            for exp_path in exps:

                # try:

                # print("*" * 50)
                # print("Evaluating model:", exp_path)

                model_path = exp_path + '/val_best.pth'
                args_path = exp_path + '/args.pkl'
                with open(args_path, 'rb') as f:
                    args = pickle.load(f)

                stats = exp_path + '/constant_metrics.pkl'
                with open(stats, 'rb') as f:
                    cm = pickle.load(f)
                print("Stats:", cm)

                #Data prep
                obs_seq_len = args.obs_seq_len
                pred_seq_len = args.pred_seq_len
                data_set = './datasets/' + args.dataset + '/'


                for class_num in ['test_0119', 'test_0120',
                                  'test_0219', 'test_0220', 'test_0221', 'test_0222', 'test_0223',
                                  'test_0319', 'test_0320', 'test_0321', 'test_0322', 'test_0323', 'test_0324',
                                  'test_0325', 'test_0326', 'test_0327', 'test_0328', 'test_0329',
                                  'test_0519', 'test_0520', 'test_0521', 'test_0522', 'test_0523', 'test_0524',
                                  'test_0525', 'test_0526', 'test_0527', 'test_0528', 'test_0529', 'test_0530',
                                  'test_0531', 'test_0532']:
                    dset_test = TrajectoryDataset(data_set + f'kpspace_test/{class_num}/',
                                                  obs_len=obs_seq_len,
                                                  pred_len=pred_seq_len,
                                                  skip=1,
                                                  norm_lap_matr=True)

                    loader_test = DataLoader(
                        dset_test,
                        batch_size=
                        1,  #This is irrelative to the args batch size parameter
                        shuffle=False,
                        num_workers=1)

                    #Defining the model

                    is_eth = args.dataset == 'eth'
                    if is_eth:
                        noise_weight = CFG["noise_weight_eth"]
                    else:
                        noise_weight = CFG["noise_weight"]

                    model = SocialImplicit(spatial_input=CFG["spatial_input"],
                                           spatial_output=CFG["spatial_output"],
                                           temporal_input=CFG["temporal_input"],
                                           temporal_output=CFG["temporal_output"],
                                           bins=CFG["bins"],
                                           noise_weight=noise_weight).cuda()
                    model.load_state_dict(torch.load(model_path))
                    model.cuda().double()
                    model.eval()

                    ade_ = 999999
                    fde_ = 999999
                    # print(f"Testing ....{class_num}!")
                    ad, fd = test(KSTEPS=KSTEPS)
                    ade_ = min(ade_, ad) * 0.75
                    fde_ = min(fde_, fd) * 0.75
                    ade_ls.append(ade_)
                    fde_ls.append(fde_)
                    exp_ls.append(exp_path)
                    # print(f"{class_num}_ADE:", ade_, f" {class_num}_FDE:", fde_)
                    EASY_RESULTS.append([class_num, round(ade_, 4), round(fde_, 4)])
                # print(EASY_RESULTS)

                t = ''
                with open('Ab_Ex_only_zerocentered_RESULTS.txt', 'w') as q:
                    for i in EASY_RESULTS:
                        for e in range(len(EASY_RESULTS[0])):
                            t = t + str(i[e]) + ' '
                        q.write(t.strip(' '))
                        q.write('\n')
                        t = ''
                print('done')