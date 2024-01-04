# from mid import MID
####################################
# from mid_kp_eth import MID
# from mid_MOT import MID
# from MID_all_eval_test import MID
# from MID_all_train import MID
# from HT21_all_train import MID
# from HT21_all_eval_test import MID
# from VSC_all_train import MID
from VSC_all_eval_test import MID
####################################
import argparse
import os
import yaml
# from pprint import pprint
from easydict import EasyDict
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    parser.add_argument('--config', default='./configs/VSC.yaml')
    parser.add_argument('--dataset', default='VSC_all_data_kpspace')
    return parser.parse_args()


# def main():
#     # parse arguments and load config
#     args = parse_args()
#     with open(args.config, encoding='utf-8') as f:
#        config = yaml.safe_load(f)
#
#     for k, v in vars(args).items():
#        config[k] = v
#     config["exp_name"] = args.config.split("/")[-1].split(".")[0]
#     config["dataset"] = args.dataset
#
#     config = EasyDict(config)
#     agent = MID(config)
#
#     if config["eval_mode"]:
#         agent.eval()
#     else:
#         agent.train()

def main():
    ####################################################################################################################
# ['test_07219', 'test_07220', 'test_07319', 'test_07320', 'test_07419', 'test_07420',
# 'test_07619', 'test_07620', 'test_07621', 'test_07622', 'test_07623',
# 'test_07624', 'test_07625', 'test_07626', 'test_07627', 'test_07628', 'test_07629',
# 'test_07630', 'test_07719', 'test_10019', 'test_10020', 'test_10021', 'test_10022',
# 'test_10023', 'test_10024', 'test_10025', 'test_10026', 'test_10027', 'test_10028',
# 'test_10029', 'test_11119', 'test_11120', 'test_11121', 'test_11122', 'test_17119',
# 'test_17120', 'test_17121', 'test_17122', 'test_17123', 'test_17124', 'test_17125',
# 'test_28619', 'test_28620', 'test_28621', 'test_34819', 'test_34820', 'test_34821',
# 'test_34822', 'test_34823', 'test_34824', 'test_34825', 'test_34826', 'test_46419',
# 'test_46420', 'test_46421', 'test_46422', 'test_46423', 'test_49219', 'test_49220',
# 'test_49221', 'test_49222', 'test_49223', 'test_49224', 'test_49225', 'test_49226',
# 'test_49227', 'test_49228', 'test_49229', 'test_49230', 'test_49231', 'test_49232',
# 'test_49233', 'test_49234', 'test_49235', 'test_49236', 'test_49237', 'test_49238',
# 'test_49239', 'test_49240', 'test_49241', 'test_49242', 'test_49243', 'test_49244',
# 'test_49245', 'test_49246', 'test_49319', 'test_49320', 'test_49321', 'test_49322',
# 'test_49324', 'test_49325', 'test_49326', 'test_49327', 'test_49328',
# 'test_49329', 'test_49330', 'test_49331', 'test_49332', 'test_49333', 'test_49334',
# 'test_49335', 'test_49336', 'test_49337', 'test_49338', 'test_49339', 'test_49340',
# 'test_49341', 'test_49342', 'test_49343', 'test_49344', 'test_49345', 'test_49346',
# 'test_49347', 'test_49348', 'test_49349', 'test_49350', 'test_49351', 'test_49352',
# 'test_49353', 'test_49419', 'test_49420', 'test_49421', 'test_49422', 'test_49423',
# 'test_49424', 'test_49425', 'test_49426', 'test_49427', 'test_49428', 'test_49429',
# 'test_49430', 'test_49431', 'test_49432', 'test_49433', 'test_49434', 'test_49435',
# 'test_49436', 'test_49437', 'test_49438', 'test_49439', 'test_49440', 'test_49441',
# 'test_49442', 'test_49443', 'test_49444', 'test_49445', 'test_49446', 'test_49447',
# 'test_49448']
    data_class = 'test_49448'
    # parse arguments and load config
    args = parse_args()
    with open(args.config, encoding='utf-8') as f:
       config = yaml.safe_load(f)

    for k, v in vars(args).items():
       config[k] = v
    config["exp_name"] = args.config.split("/")[-1].split(".")[0]
    config["dataset"] = args.dataset
    config['class_num'] = data_class.split('_')[-1]

    config = EasyDict(config)
    agent = MID(config)

    if config["eval_mode"]:
        agent.eval()
    else:
        agent.train()



if __name__ == '__main__':
    main()



