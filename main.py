import argparse
import csv

import draw as d
import torch
#from td3 import TD3Trainer
from two.td33 import TD3Trainer
#from two.DDPG import TD3Trainer
parser = argparse.ArgumentParser(description='Input n_agents and main folder')
parser.add_argument('--agents', type=int, default=9)
parser.add_argument('--folder', type=str)
parser.add_argument('--global_', type=str, default="GLOBAL")

args = parser.parse_args()

N_AGENTS = args.agents
MAIN_FOLDER = args.folder

def main():
    print(f'START AGENTS: {N_AGENTS} FOLDER: {MAIN_FOLDER}')
    torch.set_num_threads(1)
    trainer = TD3Trainer(N_AGENTS)

    f = open("file.csv", "w", encoding="gbk", newline="")
    uav= [f'uav_{i}' for i in range(N_AGENTS)]
    csv.writer(f).writerow(uav)
    for i in range(40001):
        print(f'{i+1} ')
        r = trainer.train_one_episode()
        print(r)
        if (i % 200 == 0) and i != 0 :

            print(trainer.test_one_episode())
            d.newcsv(trainer.sumway,trainer.End,trainer.rew,f,trainer.lostend,trainer.lostway,trainer.losttime)
    f.close()


if __name__ == '__main__':
    main()

