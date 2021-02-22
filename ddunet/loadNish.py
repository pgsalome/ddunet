#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 13:32:55 2021

@author: e210
"""



import os
import numpy as np
import matplotlib.pyplot as plt
import torch

inp = './input_40_En2_32samples.pt'
out = './output_40_En2_32samples.pt'

out_dir = '/home/e210/Python/ddunet/train'

ct_sample = torch.load(inp)
dose_sample = torch.load(out)

for i in range(len(ct_sample)):
    os.makedirs(os.path.join(out_dir,str(i)))
    ct = ct_sample[i,:,:,:]
    dose = dose_sample[i,:,:,:]
    ct_fn = os.path.join(out_dir,str(i)) +'/ct.pt'
    dose_fn = os.path.join(out_dir,str(i)) + '/dd.pt'
    torch.save(ct,ct_fn)
    torch.save(dose,dose_fn)

n = 16

ct1 = ct_sample[n, :, :, :]
dose1 = dose_sample[n, :, :, :]

ct1_slice = torch.sum(ct1, dim = 1)
dose1_slice = torch.sum(dose1, dim = 1)