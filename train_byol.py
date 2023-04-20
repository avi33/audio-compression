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
from utils.helper_funcs import save_sample, save_audio, plot_spectrogram

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--dataset", default="cmuarctic", type=str)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--sampling_rate", default=8000, type=int)
    parser.add_argument("--seq_len", default=8000, type=int)
    parser.add_argument('--augs_signal', nargs='+', type=str,
                        default=['amp', 'neg', 'tshift', 'pitchshift', 'cycshift', 'flip'])
    parser.add_argument('--augs_noise', nargs='+', type=str,
                        default=[])        
    '''optimizer'''
    parser.add_argument("--max_lr", default=3e-4, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)
    parser.add_argument('--ema', default=0.995, type=float)
    parser.add_argument("--amp", action='store_true', default=False)
    parser.add_argument("--use_adv", action="store_true", default=False)
    '''general'''
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--save_path", default='outputs/tmp', type=Path)
    parser.add_argument("--load_path", default=None, type=Path)
    parser.add_argument("--save_interval", default=100, type=int)    
    parser.add_argument("--log_interval", default=100, type=int)
    
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    return args

def create_dataset(args):
    if args.dataset == 'cmuarctic':
        transforms_train = args.augs_signal + args.augs_noise
        from data.cmudata import CMUDataset as Dataset        
        train_set = Dataset(root=r"/media/avi/8E56B6E056B6C86B/datasets/ARCTIC8k", 
                            mode='train', 
                            segment_length=args.seq_len, 
                            sampling_rate=args.sampling_rate, 
                            augment=transforms_train, 
                            return_orig=True)
        
        transforms_test = transforms_train.copy()

        test_set = Dataset(root=r"/media/avi/8E56B6E056B6C86B/datasets/ARCTIC8k", 
                           mode='test', 
                           segment_length=args.seq_len, 
                           sampling_rate=args.sampling_rate, 
                           augment=transforms_test, 
                           return_orig=True)
        
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
    
    ####################################
    # Data loaders                     #
    ####################################
    train_set, test_set = create_dataset(args)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=False)

    ####################################
    # Network                          #
    ####################################    
    from byol import BYOL
    from byol import Predictor
    from modules.encoder import ContentEncoder
    online_network = ContentEncoder(dim_input=1, dim_latent=1, win_len=1024, hop_len=256, n_fft=1024)
    target_network = ContentEncoder(dim_input=1, dim_latent=1, win_len=1024, hop_len=256, n_fft=1024)
    predictor_network = Predictor(256, 64)

    # Initialize BYOL module
    net = BYOL(online_network, target_network, predictor_network)
    net.to(device)
    ####################################
    # Optimizer                        #
    ####################################
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
    if args.ema is not None and args.ema > 0:
        from modules.ema import ModelEma as EMA
        ema = EMA(net, decay_per_epoch=args.ema)
        epochs_from_last_reset = 0
        decay_per_epoch_orig = args.ema
    
    ##########################
    # Resume training        #
    ##########################
    if load_root and load_root.exists():
        checkpoint = torch.load(load_root / "chkpnt.pt")
        net.load_state_dict(checkpoint['model_dict'])
        opt.load_state_dict(checkpoint['opt_dict'])
        print('checkpoints loaded')

    ##########################
    # training loop          #
    ##########################
    steps = 0
    torch.backends.cudnn.benchmark = True
    for epoch in range(1, args.n_epochs + 1):
        metric_logger = logger.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", logger.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Epoch: [{epoch}]"
        if args.ema is not None and args.ema > 0:
            if epochs_from_last_reset <= 1:  # two first epochs do ultra short-term ema
                ema.decay_per_epoch = 0.01
            else:
                ema.decay_per_epoch = decay_per_epoch_orig
            epochs_from_last_reset += 1
            # set 'decay_per_step' for the eooch
            ema.set_decay_per_step(len(train_loader))        
        
        for iterno, (x, _, xa) in  enumerate(metric_logger.log_every(train_loader, args.log_interval, header)):                    
            x = x.to(device)
            xa = xa.to(device)
            with torch.cuda.amp.autocast(enabled=scaler is not None):                                                
                z1, z2, p1, p2 = net(x, xa)
                
                loss = BYOL.compute_loss(z1, z2, p1, p2) / x.shape[0]
                
                net.zero_grad(set_to_none=True)
                
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

                #ema on target network
                net.update_target_network()

            if args.ema is not None and args.ema > 0:
                ema.update(net, steps)

            if not skip_scheduler:
                lr_scheduler.step()
                
            '''metrics'''            
            metric_logger.update(loss=loss.item())                                
            metric_logger.update(lr=opt.param_groups[0]["lr"])
            
            ######################
            # Update tensorboard #
            ######################             
            writer.add_scalar("train/loss", loss.item(), steps)
            writer.add_scalar("lr", opt.param_groups[0]["lr"], steps)
            
            steps += 1                        
            if steps % args.save_interval == 0:                
                loss_test = 0                    
                net.eval()                
                with torch.no_grad():                                        
                    for i, (x, _, xa) in enumerate(test_loader):                        
                        x = x.to(device)
                        xa = xa.to(device)
                        z1, z2, p1, p2 = net(x, xa)
                        loss_test += BYOL.compute_loss(z1, z2, p1, p2).item() / x.shape[0]
                                
                loss_test /= len(test_loader)                    

                metric_logger.update(loss_test=loss_test)
                
                writer.add_scalar("test/loss", loss_test, steps)
                                                                    
                chkpnt = {
                    'model_dict': net.state_dict(),
                    'opt_dict': opt.state_dict(),
                    'step': steps
                }
                
                torch.save(chkpnt, root / "chkpnt.pt")

                net.train()
                ###End main loop###

if __name__ == "__main__":
    train()