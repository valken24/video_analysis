import torch
from tqdm import tqdm

def test_model(model, dataloader, device, verbose):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(dataloader) if verbose else dataloader):
            X = X.to(device)
            output = rnn_decoder(cnn_encoder(X))
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            
            all_y_pred.extend(y_pred.cpu().squeeze().data.numpy().tolist())
    
    return all_y_pred
                