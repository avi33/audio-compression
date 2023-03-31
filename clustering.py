import torch
import numpy as np
from sklearn.cluster import KMeans
import argparse
from pathlib import Path
import faiss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int)
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

def get_feat(x):
    win_len = 256
    hop_len = 128
    n_fft = 256
    X = torch.stft(x, win_length=win_len, hop_length=hop_len, n_fft=n_fft, window=torch.hann_window(win_len), return_complex=True).abs()
    X = torch.log10(X + 1e-5)
    X = X.permute(1, 2, 0).contiguous().view(-1, win_len//2+1).contiguous()
    return X

def kmeans_batched(train_loader, test_loader, k):
    # Split the data into batches    
    num_epochs = 1
    ema_decay = 0.995
    batch2buffer = 64
    # Cluster each batch of data
    X = torch.empty(0)
    # kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans = faiss.Kmeans(d=129, k=30, nredo=10, verbose=True)
    for epoch in range(num_epochs):
        for i, (x, _) in enumerate(train_loader):
            x = x.squeeze(1)
            if i % 100 == 0:
                print(f"{epoch}->{i}/{len(train_loader)}")
            if epoch == 0 and i < batch2buffer:
                # Initialize the k-means algorithm using the first batch of data                
                    X_ = get_feat(x)                    
                    X = torch.cat((X, X_), dim=0)
                    continue                    
            elif epoch == 0 and i == batch2buffer:                
                kmeans.train(X)
                ema_cluster_centers = torch.from_numpy(kmeans.centroids).clone()
            else:
                X = get_feat(x)
                # Assign each data point to its nearest centroid
                distances = torch.cdist(X, ema_cluster_centers)
                nearest_indices = torch.argmin(distances, dim=1)
                # Update the centroids based on the newly assigned data points                
                for j in range(k):
                    kmeans.centroids[j] = torch.mean(X[nearest_indices == j], dim=0)

                ema_cluster_centers = ema_decay * ema_cluster_centers + (1 - ema_decay) * kmeans.centroids            
                score = torch.sum(torch.min(distances, dim=1)[0] ** 2)
                if i % 100 == 0:
                    print("score={}".format(score))

        print(torch.isnan(ema_cluster_centers).sum() == 0)
        
        score_test = 0
        for j, (x, _) in enumerate(test_loader):
            X = get_feat(x)
            distances = torch.cdist(X, ema_cluster_centers)
            score_test += torch.sum(torch.min(distances, dim=1)[0] ** 2)
            print("score={}".format(score_test/j))
        score_test /= len(test_loader)
        print("score={}".format(score_test))
    
    # Return the final cluster assignments
    return kmeans.centroids

def run():
    args = parse_args()
    from data.cmudata import CMUDataset
    train_set = CMUDataset(root=r"/media/avi/8E56B6E056B6C86B/datasets/ARCTIC8k", 
                           mode='train', 
                           segment_length=args.seq_len, 
                           sampling_rate=args.sampling_rate, 
                           augment=None,
                           trim=False)
    test_set = CMUDataset(root=r"/media/avi/8E56B6E056B6C86B/datasets/ARCTIC8k", 
                          mode='test', segment_length=args.seq_len, sampling_rate=args.sampling_rate, augment=None)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=False)
    kmeans_centers = kmeans_batched(train_loader, test_loader, k=30)    
    torch.save(kmeans_centers, "kmeans.pt")


if __name__ == "__main__":
    run()