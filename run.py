import argparse
import yaml
import pkgutil
import importlib
import sys
import numpy as np
import json
import os.path

from simulator import Simulator
from tqdm import tqdm
from env.env import Env
from alg.alg import Alg


def load_all_modules_from_dir(dirname):
    for importer, package_name, _ in pkgutil.iter_modules([dirname]):
        if package_name not in sys.modules and package_name != 'main':
            importlib.import_module(f'{dirname}.{package_name}')

load_all_modules_from_dir('env')
load_all_modules_from_dir('alg')

parser = argparse.ArgumentParser(
    description='NeurIPS 2022 experiment launcher')
parser.add_argument('--config', help='yaml config file path')
parser.add_argument('--seed', type=int, help='random seed')
parser.add_argument('--output_dir', help='output directory path')
parser.add_argument('--silent', action='store_true')
parser.add_argument('--dry_run', action='store_true')
parser.add_argument('--override')
parser.add_argument('--env_override')

args = parser.parse_args()

with open(args.config, 'r') as f:
    conf = yaml.safe_load(f)

REQUIRED_FIELDS = [
    'env',
    'alg',
]

for field in REQUIRED_FIELDS:
    if field not in conf:
        print(f'Error: A required field "{field}" not specified')
        sys.exit(1)

if args.seed is not None:
    np.random.seed(args.seed)
    conf['alg']['seed'] = args.seed

# override
if args.override:
    for pair in args.override.split(','):
        k, v = pair.split(':')
        if v == 'True':
            v = True
        elif v == 'False':
            v = False
        else:
            try:
                v = float(v)
            except:
                pass
        conf_ref = conf['alg']
        for fragment in k.split('+')[:-1]:
            conf_ref = conf_ref[fragment]
        conf_ref[k.split('+')[-1]] = v

if args.env_override:
    for pair in args.env_override.split(','):
        k, v = pair.split(':')
        if v == 'True':
            v = True
        elif v == 'False':
            v = False
        else:
            try:
                v = float(v)
            except:
                pass
        conf['env'][k] = v
        if 'env' in conf['env']:
            conf['env']['env'][k] = v

# output file
config_filename = args.config.split('/')[-1].rsplit('.', 1)[0]
if args.output_dir:
    file_name = f'{args.output_dir}/{config_filename}'
    if args.override is not None:
        file_name += f'_{args.override}'
    if args.env_override is not None:
        file_name += f'_{args.env_override}'
    if args.seed is not None:
        file_name += f'_{args.seed}'
    file_name = f'{file_name}.log'
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            c = 0
            for line in f:
                c += 1
            if c >= conf['env']['T']:
                sys.exit(0)

    f = open(file_name, 'w')
if args.dry_run:
    sys.exit(1)

env = Env.create(conf['env']['name'], conf['env'])
alg = Alg.create(conf['alg']['name'], env, conf['alg'])
T = env.T # hack

sim = Simulator(env, alg, T)
regrets = np.zeros(T)
if not args.silent:
    pbar = tqdm(sim)
else:
    pbar = sim

# print config for the record
final_conf = {
    'env': env.config,
    'alg': alg.config,
}
final_conf['env']['name'] = conf['env']['name']
final_conf['alg']['name'] = conf['alg']['name']
if args.output_dir:
    print(f'{json.dumps(final_conf)}', file=f)
if not args.silent:
    print(json.dumps(final_conf, indent=4))
for t, a, r, log in pbar:
    regret = env.regret(t, a)
    mean_reward = env.mean_reward(t, a)
    regrets[t - 1] = regret
    if not args.silent:
        pbar.set_description('regret: {:.4f}'.format(np.sum(regrets) / t))
    if args.output_dir:
        print(f'{t}\t{a}\t{r}\t{mean_reward}\t{regret}\t{json.dumps(log)}', file=f)
