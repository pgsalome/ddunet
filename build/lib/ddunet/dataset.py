  

from torch.utils.data import Dataset

import torch
import os

class unetDataset(Dataset):

    def __init__(self, list_inp='', transform = None):
        """
        Args:
            
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.list_inp = list_inp
        self.transform = transform
    def __len__(self):
        return len(self.list_inp)
    
    
    def __getitem__(self, idx):
        
        
        ct_fn = os.path.join(self.list_inp[idx],'ct.pt')
        if os.path.isfile(ct_fn):

            ct = torch.load(ct_fn).permute(1,2,0).numpy()
        
        dose_fn = os.path.join(self.list_inp[idx],'dd.pt')
        if os.path.isfile(dose_fn):
            dose = torch.load(dose_fn).permute(1,2,0).numpy()

        else:
            dose = None

               
        dim = ct.shape

        
        sample = {'image': ct, 'label': dose, 'spacing': dim, 'fn': ct_fn}
        if self.transform:
            sample = self.transform(sample)

        return sample
