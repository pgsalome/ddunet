import time
import torch
import copy
from ddunet.utils import convert_to_numpy
from skimage.metrics import  peak_signal_noise_ratio
import numpy as np

def train_model(model,dataloaders, criterion, optimizer, scheduler, num_epochs,device,dataset_sizes):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 330.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            
            running_loss = 0.0
            #running_corrects = 0
            if phase == 'train':
                scheduler.step(metrics=running_loss)
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode



            # Iterate over data.
            for step, data in enumerate(dataloaders[phase]):
#                print(step, data['image'].size(),
#                      data['label'].size())
                inputs = data['image']
                labels=data['label']
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                
                
                i, t = convert_to_numpy(preds, labels)
                t = np.squeeze(t, axis=(1,))

                #running_corrects += 

            epoch_loss = running_loss / dataset_sizes[phase]
            #epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} '.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_loss