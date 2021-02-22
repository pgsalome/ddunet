# -*- coding: utf-8 -*-

from torchvision import transforms
import torch 
from torch.utils.data import DataLoader
from unet.transforms import ToTensor, resize_3Dimage
from unet.utils import load_checkpoint
from unet.dataset import unetDataset
import matplotlib.pyplot as plt
import numpy as np
from unet.utils import resize_3Dimagenp

in_real = ['/home/e210/Python/unet/train/input/0.pt']
out_real = ['/home/e210/Python/unet/train/output/0.pt']

spatial_size = 128
nr_slices = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose(
                [resize_3Dimage(spatial_size,nr_slices),
                 ToTensor()])


model = load_checkpoint('test.pth')

test_dataset = unetDataset(list_inp=in_real , list_out=out_real, transform = data_transforms)

test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=8)
train_dataloader = DataLoader(train_dataset, batch_size = 1, shuffle=False, num_workers=1)


for step, data in enumerate(test_dataloader):
    
    inputs = data['image']
    img_name = data['fn']
    
    dose = data['label']
    print(dose.max())

    inputs = inputs.to(device)
    output = model(inputs)
    prob = output.data.cpu().numpy()

prob = np.squeeze(prob, axis=(0,1,))
prob_ = resize_3Dimagenp(prob,15,150)

dose = np.squeeze(dose.numpy(), axis=(0,1,))
dose_ = resize_3Dimagenp(dose,15,150)
#
ct = torch.load(in_real[0]).permute(1,2,0).numpy()
dose = torch.load(out_real).permute(1,2,0).numpy()

i = 25
fig, axs = plt.subplots(nrows=3, figsize=[15, 7])
axs[0].imshow(ct[:,:,i], cmap = plt.gray())
axs[0].set_aspect('auto')
axs[1].imshow(dose[:,:,i], cmap = plt.jet())
axs[1].set_aspect('auto')
axs[2].imshow(prob[:,:,i], cmap = plt.jet())
axs[2].set_aspect('auto')
plt.show()

###########################################################################

prob = prob.transpose(2,0,1)

inp = '/home/pgsalome/Downloads/Pato/input_40_En2_32samples.pt'
out = '/home/pgsalome/Downloads/Pato/output_40_En2_32samples.pt'

ct_sample = torch.load(inp)
dose_sample = torch.load(out)

ct_sample.shape, dose_sample.shape

n = 16

ct1 = ct_sample[n, :, :, :]
dose1 = dose_sample[n, :, :, :]

ct1_slice = torch.sum(ct1, dim = 1)
dose1_slice = torch.sum(dose1, dim = 1)
dose2_slice = torch.sum(torch.from_numpy(prob), dim=1)

fig, axs = plt.subplots(nrows=3, figsize=[15, 7])
axs[0].imshow(ct1_slice.T, cmap = plt.gray())
axs[0].set_aspect('auto')
axs[1].imshow(dose1_slice.T, cmap = plt.jet())
axs[1].set_aspect('auto')
axs[2].imshow(dose2_slice.T, cmap = plt.jet())
axs[2].set_aspect('auto')
plt.show()


###########################################################################









#  





