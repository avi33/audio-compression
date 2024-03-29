import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import numpy as np
import argparse
from pathlib import Path
from utils.helper_funcs import add_weight_decay
import utils.logger as logger
import copy
from utils.helper_funcs import save_sample, save_audio, accuracy

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(device=None))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
# 0MB allocation and cache
print(torch.cuda.memory_allocated()/1024**2)
print(torch.cuda.memory_cached()/1024**2)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--dataset", default="cmuarctic", type=str)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--sampling_rate", default=8000, type=int)
    parser.add_argument("--seq_len", default=8000, type=int)
    parser.add_argument('--augs_signal', nargs='+', type=str,
                        default=['amp', 'neg'])
    parser.add_argument('--augs_noise', nargs='+', type=str,
                        default=[])
    '''optimizer'''
    parser.add_argument("--max_lr", default=3e-4, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)
    parser.add_argument('--ema', default=0.9998, type=float)
    parser.add_argument("--amp", action='store_true', default=False)        
    '''general'''
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--save_path", default='outputs/tmp', type=Path)
    parser.add_argument("--load_path", default=None, type=Path)
    parser.add_argument("--save_interval", default=100, type=int)    
    parser.add_argument("--log_interval", default=100, type=int)
   
    args = parser.parse_args()
    return args

def create_dataset(args):
    if args.dataset == 'cmuarctic':
        from data.cmudata import CMUDataset as Dataset        
        train_set = Dataset(root=r"/media/avi/8E56B6E056B6C86B/datasets/ARCTIC8k", mode='train', segment_length=args.seq_len, sampling_rate=args.sampling_rate, augment=None)
        test_set = Dataset(root=r"/media/avi/8E56B6E056B6C86B/datasets/ARCTIC8k", mode='test', segment_length=args.seq_len, sampling_rate=args.sampling_rate, augment=None)
    elif args.dataset == 'vctk':
        pass
    else:
        raise ValueError("wrong dataset {}".format(args.dataset))
    return train_set, test_set


def train():
    args = parse_args()

    root = Path(args.save_path)
    load_root = Path(args.load_path) if args.load_path else None    
    root.mkdir(parents=True, exist_ok=True)       
    ####################################
    # Dump arguments and create logger #
    ####################################
    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)
    writer = SummaryWriter(str(root))

    '''data'''
    train_set, test_set = create_dataset(args)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)

    '''net'''
    win_len = 256
    hop_len = 128
    n_fft = 256
    centers = torch.load("outputs/gate/kmeans.pt")
    centers = torch.from_numpy(centers).to(device)

    from modules.gate import Gate
    net = Gate(dim_in=n_fft//2+1, n_experts=centers.shape[0])
    net.to(device)    
        
    '''losses'''
    from losses.label_smoothing_ce import LabelSmoothCrossEntropyLoss
    criterion_cls = LabelSmoothCrossEntropyLoss().to(device)
    
    '''optimizer'''
    if args.amp:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler(init_scale=2**10)
        eps = 1e-4    
    else:
        scaler = None
        eps = 1e-8
    skip_scheduler = False
    
    '''filter parameters from weight decay'''
    parameters = add_weight_decay(net, weight_decay=args.wd, skip_list=())
    opt = optim.AdamW(parameters, lr=args.max_lr, betas=(0.9, 0.99), eps=eps, weight_decay=0)    
    
    '''learning rate scheduler'''
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt,
                                                       max_lr=args.max_lr,
                                                       steps_per_epoch=len(train_loader),
                                                       epochs=args.n_epochs,
                                                       pct_start=0.1,                                                       
                                                    )
    '''EMA'''
    if args.ema is not None:
        from modules.ema import ModelEma as EMA
        ema = EMA(net, decay_per_epoch=args.ema)
        epochs_from_last_reset = 0
        decay_per_epoch_orig = args.ema    
    
    if load_root and load_root.exists():
        checkpoint = torch.load(load_root / "chkpnt.pt")
        net.load_state_dict(checkpoint['model_dict'])
        opt.load_state_dict(checkpoint['opt_dict'])
        print('checkpoints loaded')

    steps = 0
    torch.backends.cudnn.benchmark = True    

    for epoch in range(1, args.n_epochs + 1):
        metric_logger = logger.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", logger.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Epoch: [{epoch}]"
        if args.ema is not None:
            if epochs_from_last_reset <= 1:  # two first epochs do ultra short-term ema
                ema.decay_per_epoch = 0.01
            else:
                ema.decay_per_epoch = decay_per_epoch_orig
            epochs_from_last_reset += 1
            # set 'decay_per_step' for the eooch
            ema.set_decay_per_step(len(train_loader))        
        
        for iterno, (x, _) in  enumerate(metric_logger.log_every(train_loader, args.log_interval, header)):                    
            x = x.to(device)            
            X = torch.stft(x, win_length=win_len, hop_length=hop_len, n_fft=n_fft, window=torch.hann_window(win_len, device=device), return_complex=True)
            X = torch.log10(X.abs() + 1e-5)
            
            dist_mat = torch.cdist(X.transpose(2, 1).contiguous().view(-1, n_fft//2+1), centers)
            pseudo_label = dist_mat.argmin(-1)

            with torch.cuda.amp.autocast(enabled=scaler is not None):                                                
                z = net(X)
                z = z.transpose(2, 1).contiguous().view(-1, centers.shape[0])
                loss = criterion_cls(z, pseudo_label)

            if args.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
                scaler.step(opt)
                amp_scale = scaler.get_scale()
                scaler.update()
                skip_scheduler = amp_scale != scaler.get_scale()
            else:
                loss.backward()
                opt.step()

            if args.ema is not None:
                ema.update(net, steps)

            if not skip_scheduler:
                lr_scheduler.step()
            
            '''metrics'''
            acc = accuracy(z, pseudo_label, topk=(1, 3,))
            
            metric_logger.update(loss=loss.item())
            metric_logger.update(acc1=acc[0])
            metric_logger.update(acc3=acc[1])
            metric_logger.update(lr=opt.param_groups[0]["lr"])

            ######################
            # Update tensorboard #
            ######################                             
            writer.add_scalar("train/loss", loss.item(), steps)
            writer.add_scalar("train/acc1", acc[0], steps)
            writer.add_scalar("train/acc3", acc[1], steps)
            writer.add_scalar("lr", opt.param_groups[0]["lr"], steps)            

            steps += 1                        
            if steps % args.save_interval == 0:                
                loss_test = 0
                acc_test = [0, 0]
                net.eval()                
                with torch.no_grad():                                        
                    for _, (x, _) in enumerate(test_loader):                        
                        x = x.to(device)            
                        X = torch.stft(x, win_length=win_len, hop_length=hop_len, n_fft=n_fft, window=torch.hann_window(win_len, device=device), return_complex=True)
                        X = torch.log10(X.abs() + 1e-5)
                        
                        dist_mat = torch.cdist(X.transpose(2, 1).contiguous().view(-1, n_fft//2+1), centers)
                        pseudo_label = dist_mat.argmin(-1)

                        z = net(X)
                        z = z.transpose(2, 1).contiguous().view(-1, centers.shape[0])
                        loss_test += F.cross_entropy(z, pseudo_label)                        
                        acc = accuracy(z, pseudo_label, topk=(1, 3,))

                        acc_test[0] += acc[0]
                        acc_test[1] += acc[1]
                                
                loss_test /= len(test_loader)
                acc_test[0] /= len(test_loader)
                acc_test[1] /= len(test_loader)

                metric_logger.update(loss=loss.item())
                metric_logger.update(acc1_t=acc[0])
                metric_logger.update(acc3_t=acc[1])                    

                writer.add_scalar("test/loss", loss_test, steps)
                writer.add_scalar("test/acc1", acc_test[0], steps)
                writer.add_scalar("test/acc3", acc_test[1], steps)
                
                
                net.train()                
                                
                chkpnt = {
                    'model_dict': net.state_dict(),
                    'opt_dict': opt.state_dict(),
                    'step': steps,
                }
                torch.save(chkpnt, root / "chkpnt.pt")

if __name__ == "__main__":
    train()