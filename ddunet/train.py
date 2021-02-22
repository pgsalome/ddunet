
import torch
from torch.utils.data import DataLoader
from ddunet.dataset import unetDataset
from ddunet.model import ResidualUNet3D
from torch.nn import SmoothL1Loss
import torch.optim as optim
from torch.optim import lr_scheduler
from ddunet.train_model import train_model
from torchvision import  transforms
from ddunet.transforms import ToTensor, resize_3Dimage
import glob
import numpy as np

def train(config):
    
    inp = config['root_dir']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']    
    spatial_size = config['spatial_size']    
    nr_slices = config['nr_slices']        
    training_split = config['training_split']  
    num_workers = config['num_workers']
    checkpoint_dir = config['checkpoint_dir']
    mode = config['mode']
    factor = config['factor']
    patience = config['patience']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']

    
    #get image directories
    list_inp = [x for x in glob.glob(inp + '/*') ]
    
    #define transforms
    data_transforms = transforms.Compose(
                    [
                    resize_3Dimage(spatial_size,nr_slices),
                     ToTensor()])
    
    dataset = unetDataset(list_inp=list_inp, transform = data_transforms)
    
    #split validation/training
    train_size = int(training_split * len(list_inp))
    test_size = len(list_inp) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
 
    image_datasets = {"train":train_dataset,
                      "val":test_dataset 
                      }
    
    dataloaders = {x:DataLoader(image_datasets[x], batch_size,
                                                 shuffle=True, num_workers=num_workers)
                  for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    ##check  tensor
    for i_batch, sample_batched in enumerate(dataloaders['train']):
         print(i_batch, sample_batched['image'].size(),
               sample_batched['label'].size())
         break
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    model = ResidualUNet3D(in_channels=1, out_channels=1)
    model = model.to(device)
    criterion =  SmoothL1Loss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    #use 'max' if evaluation is better when higher 
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode = mode , factor=factor, patience = patience)
    
    
    
    model_ft, best_loss = train_model(model,dataloaders, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs,device,dataset_sizes)
    
    
    #store checkpoint
    checkpoint = {'model': model,
              'state_dict': model_ft.state_dict(),
              'optimizer' : optimizer_ft.state_dict()}
    
    torch.save(checkpoint, checkpoint_dir+'/'+'cp_'+str(np.round(best_loss,4))+'.pth')
    
    
