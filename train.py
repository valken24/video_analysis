import os
import time
import copy
import tqdm
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs, output_folder, tensorboard, verbose):
    cnn_encoder, rnn_decoder = model
    since = time.time()
    best_encoder_wts = copy.deepcopy(cnn_encoder.state_dict())
    best_decoder_wts = copy.deepcopy(rnn_decoder.state_dict())
    best_acc = 0.0
    best_epoch = 0
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    
    writer = SummaryWriter()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                cnn_encoder.train()  # Set model to training mode
                rnn_decoder.train()
            else:
                cnn_encoder.eval()   # Set model to evaluate mode
                rnn_decoder.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]) if verbose else dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).view(-1, )

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = rnn_decoder(cnn_encoder(inputs))
                    
                    _, preds = torch.max(outputs, 1)
                    loss = F.cross_entropy(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if tensorboard:
                writer.add_scalar('{} Epoch Loss'.format(phase), epoch_loss, epoch)
                writer.add_scalar('{} Epoch Acc'.format(phase), epoch_acc, epoch)
                
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
                
            history["%s_loss"%(phase)].append(epoch_loss)
            history["%s_acc"%(phase)].append(epoch_acc)
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_encoder_wts = copy.deepcopy(cnn_encoder.state_dict())
                best_decoder_wts = copy.deepcopy(rnn_decoder.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc Epoch {}: {:4f}'.format(best_epoch, best_acc))

    # load best model weights
    cnn_encoder.load_state_dict(best_encoder_wts)
    rnn_decoder.load_state_dict(best_decoder_wts)
    
    #save the  best model
    os.makedirs(output_folder, exist_ok=True)
    torch.save(cnn_encoder.state_dict(), os.path.join(output_folder,'encoder.pth'))
    torch.save(rnn_decoder.state_dict(), os.path.join(output_folder,'decoder.pth'))
    
    return cnn_encoder, rnn_decoder, history, best_acc
