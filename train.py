import argparse
from copy import deepcopy
import csv
import json
import os

import torch
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import gym
import ray

from models import Policy
from utils import RunningStat, Adam, compute_centered_ranks, compute_weight_decay


@ray.remote
def rollout(policy, env_name, seed=None, calc_state_stat_prob=0.01, test=False):
    save_obs = not test and np.random.random() < calc_state_stat_prob
    if save_obs:
        states = []
    env = gym.make(env_name)
    if seed is not None:
        env.seed(seed)
    state = env.reset()
    ret = 0
    timesteps = 0
    done = False
    while not done:
        if save_obs:
            states.append(state)
        with torch.no_grad():
            action = policy.act(state)
        state, reward, done, _ = env.step(action)
        ret += reward
        timesteps += 1
        if done:
            break
    if test:
        return ret
    if save_obs:
        return ret, timesteps, np.array(states)
    return ret, timesteps, None


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='Humanoid-v2')
parser.add_argument('--num_parallel', type=int, default=64)
parser.add_argument('--episodes_per_batch', type=int, default=10000)
parser.add_argument('--timesteps_per_batch', type=int, default=100000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.005)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--noise_std', type=float, default=0.02)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--outdir', type=str, default='result')
args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)

with open(args.outdir + "/params.json", mode="w") as f:
    json.dump(args.__dict__, f, indent=4)


def run():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gym.logger.set_level(40)
    env = gym.make(args.env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    state_stat = RunningStat(
        env.observation_space.shape,
        eps=1e-2
    )
    action_space = env.action_space
    policy = Policy(state_size, action_size, args.hidden_size, action_space.low, action_space.high)
    num_params = policy.num_params
    optim = Adam(num_params, args.lr)

    ray.init(num_cpus=args.num_parallel)

    return_list = []
    for epoch in range(100000):
        #####################################
        ### Rollout and Update State Stat ###
        #####################################

        policy.set_state_stat(state_stat.mean, state_stat.std)
        
        # set diff params (mirror sampling)
        assert args.episodes_per_batch % 2 == 0
        diff_params = torch.empty((args.episodes_per_batch, num_params), dtype=torch.float)
        diff_params_pos = torch.randn(args.episodes_per_batch//2, num_params) * args.noise_std
        diff_params[::2]= diff_params_pos
        diff_params[1::2] = -diff_params_pos

        rets = []
        num_episodes_popped = 0
        num_timesteps_popped = 0
        while num_episodes_popped < args.episodes_per_batch \
                and num_timesteps_popped < args.timesteps_per_batch:
                #or num_timesteps_popped < args.timesteps_per_batch:
            results = []
            for i in range(min(args.episodes_per_batch, 500)):
                # set policy
                randomized_policy = deepcopy(policy)
                randomized_policy.add_params(diff_params[num_episodes_popped+i])
                # rollout
                results.append(rollout.remote(randomized_policy, args.env_name, seed=np.random.randint(0,10000000)))
            
            for result in results:
                ret, timesteps, states = ray.get(result)
                rets.append(ret)
                # update state stat
                if states is not None:
                    state_stat.increment(states.sum(axis=0), np.square(states).sum(axis=0), states.shape[0])
                
                num_timesteps_popped += timesteps
                num_episodes_popped += 1
        rets = np.array(rets, dtype=np.float32)
        diff_params = diff_params[:num_episodes_popped]
        
        best_policy_idx = np.argmax(rets)
        best_policy = deepcopy(policy)
        best_policy.add_params(diff_params[best_policy_idx])
        best_rets = [rollout.remote(best_policy, args.env_name, seed=np.random.randint(0,10000000), calc_state_stat_prob=0.0, test=True) for _ in range(10)]
        best_rets = np.average(ray.get(best_rets))
        
        print('epoch:', epoch, 'mean:', np.average(rets), 'max:', np.max(rets), 'best:', best_rets)
        with open(args.outdir + '/return.csv', 'w') as f:
            return_list.append([epoch, np.max(rets), np.average(rets), best_rets])
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(return_list)

            plt.figure()
            sns.lineplot(data=np.array(return_list)[:,1:])
            plt.savefig(args.outdir + '/return.png')
            plt.close('all')
        

        #############
        ### Train ###
        #############

        fitness = compute_centered_ranks(rets).reshape(-1,1)
        if args.weight_decay > 0:
            #l2_decay = args.weight_decay * ((policy.get_params() + diff_params)**2).mean(dim=1, keepdim=True).numpy()
            l1_decay = args.weight_decay * (policy.get_params() + diff_params).mean(dim=1, keepdim=True).numpy()
            fitness += l1_decay
        grad = (fitness * diff_params.numpy()).mean(axis=0)
        policy = optim.update(policy, -grad)


if __name__ == '__main__':
    run()