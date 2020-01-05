import os
import pickle
import numpy as np
import pandas as pd
#------------------------------#
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
#------------------------------#
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#------------------------------#
from test import *
from utils import *
from train import *
#------------------------------#
import argparse
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ACTION', type=str, help="Action train or test.")
    parser.add_argument('--data', type=str, default='./data', help='Path of data train')
    parser.add_argument('--epochs', type=int, default=15, help="Number of epochs.")  
    parser.add_argument('--batch_size', type=int, default=10, help="Size of each image batch")
    parser.add_argument('--checkpoint', type=str, default='weights', help='Directory where weights are saved')
    parser.add_argument('--output', type=str, default='outputs', help='Directory where output test results are saved')
    parser.add_argument("--n_cpu", type=int, default=4, help="Number of cpu threads to use during batch generation")
    parser.add_argument('--encoder_weights', type=str, default=None, help='Enconder starts from checkpoint model')
    parser.add_argument('--decoder_weights', type=str, default=None, help='Decoder starts from checkpoint model')
    parser.add_argument("--cuda", action='store_true', help="Use cuda or cpu")
    parser.add_argument("--tensorboard", action='store_true', help="Use tensorboard in training.")
    parser.add_argument("--verbose", action='store_true', help="Show training %")

    return parser.parse_args()

def run():
    # Load arguments
    args = parse_args()
    
    # Set path
    data_path = args.data
    weights_dir = args.checkpoint
    output_dir = args.output
    label_file = './action_names.txt'
        
    # EncoderCNN architecture
    CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
    CNN_embed_dim = 512         # latent dim extracted by 2D CNN
    res_size = 224              # ResNet image size
    dropout_p = 0.0             # dropout probability

    # DecoderRNN architecture
    RNN_hidden_layers = 3
    RNN_hidden_nodes = 512
    RNN_FC_dim = 256

    # training parameters
    k = 4                            # number of target category
    criterion = nn.CrossEntropyLoss()
    epochs = args.epochs             # training epochs
    batch_size = args.batch_size  
    learning_rate = 1e-5
    log_interval = 10           # interval for displaying training info
    workers = args.n_cpu

    # Select which frame to begin & end in videos
    begin_frame, end_frame, skip_frame = 1, 29, 1
    
    # Detect devices
    if args.cuda:  
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("\nEnabled CUDA: [{} GPU]\n".format(torch.cuda.device_count()))
        else:
            device = torch.device("cpu")
            print("Cuda not available, Enabling CPU.\n")
    
    else:
        device = torch.device("cpu")
        print("Enabled CPU.\n")
    
    f = open(label_file, 'r')
    
    action_names = []
    for x in f:
        action_names.append(x.replace("\n", ""))
    
    # convert labels -> category
    le = LabelEncoder()
    le.fit(action_names)

    # show how many classes there are
    list(le.classes_)

    # convert category -> 1-hot
    action_category = le.transform(action_names).reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(action_category)

    actions = []
    fnames = os.listdir(data_path)

    all_names = []
    for f in fnames:
        loc1 = f.find('v_')
        loc2 = f.find('_g')
        actions.append(f[(loc1 + 2): loc2])
        all_names.append(f)
    
    # list all data files
    all_X_list = all_names                  # all video file names
    all_y_list = labels2cat(le, actions)    # all video labels
    
    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()
    
    ## ------------------- DATALOADER FOR TRAIN OR TEST ------------------ ##
    if args.ACTION == 'train':
        print("[Training]")
        # train, test split
        # Data loading parameters
        params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True}

        train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.25, random_state=42)

        transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

        train_set, valid_set = Dataset_CRNN(data_path, train_list, train_label, selected_frames, transform=transform), \
                               Dataset_CRNN(data_path, test_list, test_label, selected_frames, transform=transform)

        train_loader = data.DataLoader(train_set, **params)
        valid_loader = data.DataLoader(valid_set, **params)# train, test split
        
        dataloaders = {'train':train_loader, 'val':valid_loader}
        dataset_sizes = {'train':len(train_set), 'val':len(valid_set)}
  
        
    elif args.ACTION == 'test':
        print("\n[Testing]")
        transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
 
        all_data_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': workers, 'pin_memory': True}
        all_data_loader = data.DataLoader(Dataset_CRNN(data_path, all_X_list, all_y_list, selected_frames, transform=transform), **all_data_params)

    # Create model
    cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
    rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                            h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)
    
    # load weights
    cnn_encoder, rnn_decoder = load_weights([cnn_encoder, rnn_decoder], args.encoder_weights, args.decoder_weights)
    
    ## ------------------- ACTIONS FOR TRAIN OR TEST ------------------ ##
    if args.ACTION == 'train':
        crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
                    list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
                    list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())

        # Optimizer
        optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)
        
        cnn_encoder, rnn_decoder, curr_history, curr_best = train_model([cnn_encoder, rnn_decoder], criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=epochs, output_folder=weights_dir, tensorboard=args.tensorboard, verbose=args.verbose)
        file = os.path.join(output_dir,"%s_%.4facc_%ep.csv"%('CRNN', curr_best, epochs))
        save_history(file, curr_history)
       
    elif args.ACTION == 'test':
        
        all_y_pred = test_model([cnn_encoder, rnn_decoder], all_data_loader, device, args.verbose)
        
        df = pd.DataFrame(data={'filename': fnames, 'y': cat2labels(le, all_y_list), 'y_pred': cat2labels(le, all_y_pred)})
        
        # all predictions
        df.to_csv(os.path.join(output_dir, "videos_prediction.csv"), index = None, header=True)  # save pandas dataframe
        
        # wrong predictions
        a = df['filename'][df['y_pred']!=df['y']].tolist()
        b = df['y'][df['y_pred']!=df['y']].tolist()
        c = df['y_pred'][df['y_pred']!=df['y']].tolist()

        df_mistakes = pd.DataFrame(data={'filename':a, 'y':b, 'y_pred':c})
        df_mistakes.to_csv(os.path.join(output_dir, 'wrong_predictions.csv'), index = None, header=True)
        
        # correct predictions
        correct_predict = ~df['filename'].isin(df_mistakes['filename'])

        df_correct = df[correct_predict]
        df_correct.to_csv(os.path.join(output_dir, 'correct_predictions.csv'), index = None, header=True)
        
        # scores
        N = len(df)
        N_correct = len(df_correct)
        N_mistake = len(df_mistakes)
        
        print()
        print("-" * 10)
        print('total videos: {}'.format(N))
        print('Accuracy: {:.2f}%'.format(N_correct/N * 100))
    
        
    
    
if __name__ == '__main__':
    args = parse_args()
    if args.ACTION == 'train' or args.ACTION == 'test':
        run()
    else:
        raise ValueError("Only action of 'train' or 'test' supported.")
        
    
    