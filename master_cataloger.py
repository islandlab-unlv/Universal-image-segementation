"""Script to make a master catalog of 2D material thicknesses

    Usage:
        - Set in_file (line ) to file location of the npz catalog for desired material/substrate.
        - Set out_file (line ) for how to save the master catalog for desired material/substrate.
        - Set layers_id (line ) to the number of layers you can fit to."""

import numpy as np
from read_npz import npz2dict

in_file = '.\\Monolayer Search\\Graphene_on_SiO2_catalog.npz'
out_file = '.\\Monolayer Search\\Graphene_on_SiO2_master_catalog.npz'
in_file_dict = npz2dict(in_file)


flake_group_dict = {}
for key in in_file_dict:
    flake_name = key[:(np.array([c for c in key])==' ').nonzero()[0][1]] ## s is a string, returns everything before the second space
    if not flake_name in flake_group_dict:
        flake_group_dict[flake_name] = []
    flake_group_dict[flake_name].append(key)

## Dictionaries to categorize the data
weight_dict = {}
blue_mean_dict = {}
green_mean_dict = {}
red_mean_dict = {}
cov_dict = {}

layers_id = 5 ## The number of layers to test for
for tt in range(layers_id): ## Each layer contains a list, which will be populated with the training data for that layer
    weight_dict[tt] = []
    blue_mean_dict[tt] = []
    green_mean_dict[tt] = []
    red_mean_dict[tt] = []
    cov_dict[tt] = []

missing_track = np.zeros(layers_id) ## Failsafe for layers that did not get trained
for ff in flake_group_dict:
    for tt in range(layers_id):
        try:
            weight_dict[tt].append(in_file_dict[flake_group_dict[ff][0]][(in_file_dict[flake_group_dict[ff][5]]==tt).nonzero()[0][0]])
            blue_mean_dict[tt].append(in_file_dict[flake_group_dict[ff][1]][(in_file_dict[flake_group_dict[ff][5]]==tt).nonzero()[0][0]])
            green_mean_dict[tt].append(in_file_dict[flake_group_dict[ff][2]][(in_file_dict[flake_group_dict[ff][5]]==tt).nonzero()[0][0]])
            red_mean_dict[tt].append(in_file_dict[flake_group_dict[ff][3]][(in_file_dict[flake_group_dict[ff][5]]==tt).nonzero()[0][0]])
            cov_dict[tt].append(in_file_dict[flake_group_dict[ff][4]][(in_file_dict[flake_group_dict[ff][5]]==tt).nonzero()[0][0]])
        except IndexError:
            missing_track[tt] += 1
            print(f'No layer {tt} information for {ff}.')
            continue
if np.any(missing_track == len(flake_group_dict)):
    layer_rej = (missing_track == len(flake_group_dict)).nonzero()[0]
    for key in layer_rej:
        weight_dict.pop(key)
        blue_mean_dict.pop(key)
        green_mean_dict.pop(key)
        red_mean_dict.pop(key)
        cov_dict.pop(key)

master_weights = {}
master_blue_mean = {}
master_green_mean = {}
master_red_mean = {}
master_cov = {}

for key in weight_dict:
    master_weights[f'weights-{key}'] = np.mean(weight_dict[key])
    master_blue_mean[f'blue mean-{key}'] = np.mean(blue_mean_dict[key])
    master_green_mean[f'green mean-{key}'] = np.mean(green_mean_dict[key])
    master_red_mean[f'red mean-{key}'] = np.mean(red_mean_dict[key])
    master_cov[f'covariance-{key}'] = np.mean(cov_dict[key], axis=0)

with open(out_file, 'wb') as f:
    np.savez(f, **master_weights, **master_blue_mean, **master_green_mean,
             **master_red_mean, **master_cov)

print(master_weights)
print(master_blue_mean)
print(master_green_mean)
print(master_red_mean)
print(master_cov)
print(f'\"{out_file}\" updated.')
