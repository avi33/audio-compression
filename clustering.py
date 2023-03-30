import torch
import numpy as np
from sklearn.cluster import KMeans
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--dataset", default="cmuarctic", type=str)    
    parser.add_argument("--sampling_rate", default=8000, type=int)
    parser.add_argument("--seq_len", default=8000, type=int)
    '''debug'''
    parser.add_argument("--save_path", default='outputs/tmp', type=Path)
    parser.add_argument("--load_path", default=None, type=Path)
    parser.add_argument("--save_interval", default=100, type=int)    
    parser.add_argument("--log_interval", default=100, type=int)
    args = parser.parse_args()
    return args

def kmeans_batched(train_loader, test_loader, k):
    # Split the data into batches
    win_len = 256
    hop_len = 128
    n_fft = 256
    num_epochs = 10
    alpha = 0.8
    # Cluster each batch of data
    for epoch in range(num_epochs):
        for i, (x, _) in enumerate(train_loader):
            if i % 10 == 0:
                print(f"{i}/{len(train_loader)}")
            X = torch.stft(x.squeeze(1), win_length=win_len, hop_length=hop_len, n_fft=n_fft, window=torch.hann_window(win_len), return_complex=True).abs()
            X = 10*torch.log10(X + 1e-5)
            X = X.permute(1, 2, 0).contiguous().view(-1, win_len//2+1).contiguous()
            if i == 0:
                # Initialize the k-means algorithm using the first batch of data            
                kmeans = KMeans(n_clusters=k, init='k-means++')
                kmeans.fit(X)
            else:
                # Assign each data point to its nearest centroid
                distances = torch.cdist(X, torch.Tensor(kmeans.cluster_centers_))                
                nearest_indices = torch.argmin(distances, dim=1)
            
                # Update the centroids based on the newly assigned data points                
                for j in range(k):
                    new_centroid = torch.mean(X[nearest_indices == j], dim=0)
                    if epoch > 0 and torch.isnan(torch.from_numpy(kmeans.cluster_centers_)).sum() == 0:
                        kmeans.cluster_centers_[j] = alpha * new_centroid + (1 - alpha) * kmeans.cluster_centers_[j]
                    else:
                        kmeans.cluster_centers_[j] = new_centroid
                    # kmeans.cluster_centers_[j] = torch.mean(X[nearest_indices == j], dim=0)
            
        for j, (x, _) in enumerate(test_loader):
            X = torch.stft(x.squeeze(1), win_length=win_len, hop_length=hop_len, n_fft=n_fft, window=torch.hann_window(win_len), return_complex=True).abs()
            X = 10*torch.log10(X + 1e-5)
            X = X.permute(1, 2, 0).contiguous().view(-1, win_len//2+1).contiguous()
            # kmeans.predict(X)
            inertia = -kmeans.score(X) / X.shape[0]
            print(f"intertia:{inertia}")
    
    # Return the final cluster assignments
    return kmeans

def run():
    args = parse_args()
    from data.cmudata import CMUDataset
    train_set = CMUDataset(root=r"/media/avi/8E56B6E056B6C86B/datasets/ARCTIC8k", mode='train', segment_length=args.seq_len, sampling_rate=args.sampling_rate, augment=None)
    test_set = CMUDataset(root=r"/media/avi/8E56B6E056B6C86B/datasets/ARCTIC8k", mode='test', segment_length=args.seq_len, sampling_rate=args.sampling_rate, augment=None)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=False)
    kmeans = kmeans_batched(train_loader, test_loader, k=10)    



if __name__ == "__main__":
    run()