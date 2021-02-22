
from skimage.metrics import  peak_signal_noise_ratio
import torch
import itertools
import numpy as np
import yaml

def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]

def convert_to_numpy(*inputs):
    """
    Coverts input tensors to numpy ndarrays

    Args:
        inputs (iteable of torch.Tensor): torch tensor

    Returns:
        tuple of ndarrays
    """

    def _to_numpy(i):
        assert isinstance(i, torch.Tensor), "Expected input to be torch.Tensor"
        return i.detach().cpu().numpy()

    return (_to_numpy(i) for i in inputs)

class PSNR:
    """
    Computes Peak Signal to Noise Ratio. Use e.g. as an eval metric for denoising task
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, input, target):
        input, target = convert_to_numpy(input, target)
        return peak_signal_noise_ratio(target, input)
    

def load_checkpoint(filepath):
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    checkpoint = torch.load(filepath, map_location=map_location)
    
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model


def resize_3Dimagenp(image,img_px_size,slice_nr):
    
    s = image.shape
    new_size_x = img_px_size
    new_size_y = img_px_size
    new_size_z = slice_nr
    delta_x = s[0]/new_size_x
    delta_y = s[1]/new_size_y
    delta_z = s[2]/new_size_z
    new_data_ct = np.zeros((new_size_x,new_size_y,new_size_z))
    for x, y, z in itertools.product(range(new_size_x),
                             range(new_size_y),
                             range(new_size_z)):
        
        new_data_ct[x][y][z] = image[int(x*delta_x)][int(y*delta_y)][int(z*delta_z)]

    
    return new_data_ct

def load_config(config_file):
    return yaml.safe_load(open(config_file, 'r'))