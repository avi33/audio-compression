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
from utils.helper_funcs import save_sample, flatten_list


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--dataset", default="cmuarctic", type=str)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--sampling_rate", default=8000, type=int)
    parser.add_argument("--seq_len", default=4000, type=int)
    '''net'''
    parser.add_argument("--net_type", default="cnn", type=str)
    '''optimizer'''
    parser.add_argument("--max_lr", default=3e-4, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)
    parser.add_argument('--ema', default=0.995, type=float)
    parser.add_argument("--amp", action='store_true', default=False)
    '''loss'''
    parser.add_argument("--loss_type", default="label_smooth", type=str)
    '''debug'''
    parser.add_argument("--save_path", default='outputs/tmp', type=Path)
    parser.add_argument("--load_path", default=None, type=Path)
    parser.add_argument("--save_interval", default=100, type=int)    
    parser.add_argument("--log_interval", default=100, type=int)
    
    '''quantization'''
    parser.add_argument("--quant", action="store_true", default=True)
    
    args = parser.parse_args()
    return args

def create_dataset(args):
    if args.dataset == 'cmuarctic':
        from data.cmudata import CMUDataset as Dataset        
        train_set = Dataset(root=r"/media/avi/8E56B6E056B6C86B/datasets/ARCTIC8k", mode='train', segment_length=args.seq_len, sampling_rate=args.sampling_rate, augment=None)
        test_set = Dataset(root=r"/media/avi/8E56B6E056B6C86B/datasets/ARCTIC8k", mode='test', segment_length=args.seq_len, sampling_rate=args.sampling_rate, augment=None)

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
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    '''net'''
    from modules.soundsrteam import SoundStream
    netG = SoundStream(C=32, D=32, n_q=8, codebook_size=10, factors=None)
    netG.to(device)    
    from modules.msstftd import MultiScaleSTFTDiscriminator
    netD = MultiScaleSTFTDiscriminator(filters=32).to(device)
    '''losses'''
    from losses.mel_reconstruction_loss import SpectralReconstructionLoss
    criterion_rec_mel = SpectralReconstructionLoss(sr=args.sampling_rate, reduction='mean', device=device)
    '''optimizer'''
    if args.amp:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler(init_scale=2**10)
        eps = 1e-4
    else:
        scaler = None
        eps = 1e-8
    
    parametersG = add_weight_decay(netG, weight_decay=args.wd, skip_list=())
    optG = optim.AdamW(parametersG, lr=args.max_lr, betas=(0.9, 0.99), eps=eps, weight_decay=0)

    optD = optim.AdamW(netD.parameters(), lr=args.max_lr, betas=(0.9, 0.99), eps=eps, weight_decay=0)
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt,
    #                                                    max_lr=args.max_lr,
    #                                                    steps_per_epoch=len(train_loader),
    #                                                    epochs=args.n_epochs,
    #                                                    pct_start=0.1,                                                       
    #                                                 )        
    if args.ema is not None:
        from modules.ema import ModelEma as EMA
        ema = EMA(netG, decay_per_epoch=args.ema)
        epochs_from_last_reset = 0
        decay_per_epoch_orig = args.ema
        

    torch.backends.cudnn.benchmark = True
    acc_test = 0
    steps = 0        
    skip_scheduler = False

    ##########################
    # Dumping original audio #
    ##########################    
    test_audio = []
    for i, (x_t, _) in enumerate(test_set):
        save_sample(root / ("original_%d.wav" % i), args.sampling_rate, x_t)
        writer.add_audio("original/sample_%d.wav" % i, x_t, 0, sample_rate=args.sampling_rate)
        test_audio.append(test_audio)
        if i > 10:
            break

    if load_root and load_root.exists():
        checkpoint = torch.load(load_root / "chkpnt.pt")
        netG.load_state_dict(checkpoint['G_model_dict'])
        optG.load_state_dict(checkpoint['G_opt_dict'])
        netD.load_state_dict(checkpoint['D_model_dict'])
        optD.load_state_dict(checkpoint['D_opt_dict'])        
        print('checkpoints loaded')

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
            with torch.cuda.amp.autocast(enabled=scaler is not None):                                                
                x_est = netG(x)
                #######################
                # Train Discriminator #
                #######################
                netD.zero_grad(set_to_none=True)
                
                logits_D_fake, features_D_fake = netD(x_est.detach())
                logits_D_real, features_D_real = netD(x)
                
                loss_D = 0
                for i, scale in enumerate(logits_D_fake):                    
                    loss_D += F.relu(1 + scale).mean()

                for i, scale in enumerate(logits_D_real):                    
                    loss_D += F.relu(1 - scale).mean()
                
                loss_D.backward()
                optD.step()
                                                              
                netG.zero_grad(set_to_none=True)                                

                '''generator'''
                logits_D_fake, features_D_fake = netD(x_est)

                loss_rec_time = F.mse_loss(x_est, x, reduction="mean")
                loss_rec_mel = criterion_rec_mel(x_est, x)
                loss_fm = 0
                loss_adv = 0
                for i, scale in enumerate(logits_D_fake):
                    loss_adv += -scale.mean()
                for i in range(len(features_D_fake)):
                    for j in range(len(features_D_fake[0])):
                        loss_fm += F.l1_loss(features_D_fake[i][j], features_D_real[i][j].detach(), reduction="mean") / features_D_real[i][j].detach().abs().mean()
                
                loss_G = loss_rec_time + loss_fm + loss_adv + loss_rec_mel

            if args.amp:
                scaler.scale(loss_G).backward()
                scaler.unscale_(optG)
                torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1)
                scaler.step(optG)
                amp_scale = scaler.get_scale()
                scaler.update()
                skip_scheduler = amp_scale != scaler.get_scale()
            else:
                loss_G.backward()
                optG.step()

            if args.ema is not None:
                ema.update(netG, steps)

            # if not skip_scheduler:
            #     lr_scheduler.step()
            
            '''metrics'''            
            metric_logger.update(lossD=loss_D.item())
            metric_logger.update(lossG=loss_G.item())
            metric_logger.update(loss_rec_time=loss_rec_time.item())
            metric_logger.update(loss_rec_mel=loss_rec_mel.item())
            metric_logger.update(loss_fm=loss_fm.item())
            metric_logger.update(lr=optG.param_groups[0]["lr"])


            ######################
            # Update tensorboard #
            ######################             
            writer.add_scalar("train/lossD", loss_D.item(), steps)
            writer.add_scalar("train/lossG", loss_G.item(), steps)
            writer.add_scalar("train/loss_rec_time", loss_rec_time.item(), steps)
            writer.add_scalar("train/loss_rec_mel", loss_rec_mel.item(), steps)
            writer.add_scalar("train/loss_fm", loss_fm.item(), steps)
            writer.add_scalar("lr", optG.param_groups[0]["lr"], steps)            

            steps += 1                        
            if steps % args.save_interval == 0:                
                loss_rec_time_test = 0
                loss_rec_mel_test = 0
                netG.eval()                
                with torch.no_grad():                                        
                    for i, (x, _) in enumerate(test_loader):                        
                        x = x.to(device)                        
                        y_est = netG(x)
                        loss_rec_time_test += F.mse_loss(x_est, x, reduction="mean")
                        loss_rec_mel_test += criterion_rec_mel(x_est, x)
                                
                loss_rec_time_test /= len(test_loader)
                loss_rec_mel_test /= len(test_loader)

                writer.add_scalar("test/loss_rec_time", loss_rec_time_test.item(), steps)
                writer.add_scalar("test/loss_rec_mel", loss_rec_mel_test.item(), steps)                

                metric_logger.update(loss_rec_time=loss_rec_time.item())
                metric_logger.update(loss_rec_mel=loss_rec_mel.item())
                
                netG.train()                
                                
                chkpnt = {
                    'G_model_dict': netG.state_dict(),
                    'G_opt_dict': optG.state_dict(),
                    'step': steps,
                    'D_model_dict': netD.state_dict(),
                    'D_opt_dict': optD.state_dict(),                    
                }
                torch.save(chkpnt, root / "chkpnt.pt")

if __name__ == "__main__":
    train()