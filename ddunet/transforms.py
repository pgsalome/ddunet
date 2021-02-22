import numpy as np
import torch
import itertools

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if sample is not None:
            image, label,spacing,fn = sample['image'], sample['label'], sample['spacing'], sample['fn']
            # torch.from_numpy(np.float32(image))

            if label is not None:
                label = torch.from_numpy(np.float32(label)).unsqueeze(dim=0)
            return {'image': torch.from_numpy(np.float32(image)).unsqueeze(dim=0),
                    'label': label,
                    'spacing':spacing,
                    'fn':fn}


    
class ZscoreNormalization(object):
    """ put data in range of 0 to 1 """
   
    def __call__(self,sample):
        
        if sample is not None:
            image, label,spacing,fn = sample['image'], sample['label'], sample['spacing'], sample['fn']
            image -= image.mean() 
            image /= image.std() 
                
            return {'image': image, 'label': label, 'spacing':spacing, 'fn':fn}
        else:
            return None

class resize_3Dimage:
    """ Args: img_px_size slices resolution(cubic)
              slice_nr Nr of slices """
    
    def __init__(self,img_px_size,slice_nr):
        self.img_px_size=img_px_size
        self.slice_nr=slice_nr
    
    def __call__(self,sample):
        
        image, dose, spacing, fn = sample['image'], sample['label'], sample['spacing'], sample['fn']

        s = image.shape
    
        new_size_x = self.img_px_size
        new_size_y = self.img_px_size
        new_size_z = self.slice_nr
        delta_x = s[0]/new_size_x
        delta_y = s[1]/new_size_y
        delta_z = s[2]/new_size_z
        new_data_ct = np.zeros((new_size_x,new_size_y,new_size_z))
        new_data_dose = np.zeros((new_size_x,new_size_y,new_size_z))
        for x, y, z in itertools.product(range(new_size_x),
                                 range(new_size_y),
                                 range(new_size_z)):
            
            new_data_ct[x][y][z] = image[int(x*delta_x)][int(y*delta_y)][int(z*delta_z)]
            if dose is not None:
                new_data_dose[x][y][z] = dose[int(x*delta_x)][int(y*delta_y)][int(z*delta_z)]
        
    
        return {'image': new_data_ct, 'label': new_data_dose, 'spacing':spacing, 'fn':fn}
    
    
    