from typing import Any
import numpy as np
import torch
import torchaudio
import random
import torch.nn.functional as F
from utils.helper_funcs import AugBasic
from augs.tdpsola import change_rhytm_from_curve


class RandomTimeStretch:
    def __init__(self, p=0.5, fs=None):
        self.p = p
        self.fs = fs
        self.rhytm_modes = ['constant', 'parabola', 'down', 'up', 'question', 'stress']
    
    def __call__(self, sample):
        if random.random()<self.p:
            idx = random.randint(0, len(self.rhytm_modes)-1)
            sample = change_rhytm_from_curve(sample.numpy(), mode=self.rhytm_modes[idx], sr=self.fs)
            sample = torch.from_numpy(sample).float()
        return sample
        

class RandomTimeShift:
    def __init__(self, p=0.5, max_time_shift=None):
        self.p = p
        self.max_time_shift = max_time_shift

    def __call__(self, sample):
        if random.random() < self.p:
            if self.max_time_shift is None:
                self.max_time_shift = sample.shape[-1] // 10
            int_d = 2*random.randint(0, self.max_time_shift)-self.max_time_shift
            frac_d = np.round(100*(random.random()-0.5)) / 100
            if int_d + frac_d == 0:
                return sample
            if int_d > 0:
                pad = torch.zeros(int_d, dtype=sample.dtype)
                sample = torch.cat((pad, sample[:-int_d]), dim=-1)
            elif int_d < 0:
                pad = torch.zeros(-int_d, dtype=sample.dtype)
                sample = torch.cat((sample[-int_d:], pad), dim=-1)
            else:
                pass
            if frac_d == 0:
                return sample
            n = sample.shape[-1]
            dw = 2 * np.pi / n
            if n % 2 == 1:
                wp = torch.arange(0, np.pi, dw)
                wn = torch.arange(-dw, -np.pi, -dw).flip(dims=(-1,))
            else:
                wp = torch.arange(0, np.pi, dw)
                wn = torch.arange(-dw, -np.pi - dw, -dw).flip(dims=(-1,))
            w = torch.cat((wp, wn), dim=-1)
            phi = frac_d * w
            sample = torch.fft.ifft(torch.fft.fft(sample) * torch.exp(-1j * phi)).real
        return sample


class RandomTimeMasking:
    def __init__(self, p=0.5, n_mask=None):
        self.n_mask = n_mask
        self.p = p

    def __call__(self, sample):
        if self.n_mask is None:
            self.n_mask = int(0.05 * sample.shape[-1])
        if random.random() < self.p:
            max_start = sample.size(-1) - self.n_mask
            idx_rand = random.randint(0, max_start)
            sample[idx_rand:idx_rand + self.n_mask] = torch.randn(self.n_mask) * 1e-6
        return sample


class RandomMuLawCompression:
    def __init__(self, p=0.5, n_channels=256):
        self.n_channels = n_channels
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            e = torchaudio.functional.mu_law_encoding(sample, self.n_channels)
            sample = torchaudio.functional.mu_law_decoding(e, self.n_channels)
        return sample


class RandomAmp:
    def __init__(self, low, high, p=0.5):
        self.low = low
        self.high = high
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            amp = torch.FloatTensor(1).uniform_(self.low, self.high)
            sample.mul_(amp)
        return sample


class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample.data = torch.flip(sample.data, dims=[-1, ])
        return sample


class RandomAdd180Phase:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample.mul_(-1)
        return sample


class RandomAdditiveWhiteGN:
    def __init__(self, p=0.5, snr_db=30):
        self.snr_db = snr_db
        self.min_snr_db = 30
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            s = torch.sqrt(torch.mean(sample ** 2))
            snr_db = self.min_snr_db + torch.rand(1) * (self.snr_db - self.min_snr_db)
            sgm = s * 10 ** (-snr_db / 20.)
            w = torch.randn_like(sample).mul_(sgm)
            sample.add_(w)
        return sample


class RandomAdditiveUN:
    def __init__(self, snr_db=35, p=0.5):
        self.snr_db = snr_db
        self.min_snr_db = 30
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            s = torch.sqrt(torch.mean(sample ** 2))
            snr_db = self.min_snr_db + torch.rand(1) * (self.snr_db - self.min_snr_db)
            sgm = s * 10 ** (-snr_db / 20.) * np.sqrt(3)
            w = torch.rand_like(sample).mul_(2 * sgm).add_(-sgm)
            sample.add_(w)
        return sample


class RandomAdditivePinkGN:
    def __init__(self, snr_db=35, p=0.5):
        self.snr_db = snr_db
        self.min_snr_db = 30
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            s = torch.sqrt(torch.mean(sample ** 2))
            n = sample.shape[-1]
            w = torch.randn(n)
            nn = n // 2 + 1
            k = torch.arange(1, nn + 1, 1).float()
            W = torch.fft.fft(w)
            W = W[:nn] / k.sqrt()
            W = torch.cat((W, W.flip(dims=(-1,))[1:-1].conj()), dim=-1)
            w = torch.fft.ifft(W).real
            w.add_(w.mean()).div_(w.std())
            snr_db = self.min_snr_db + torch.rand(1) * (self.snr_db - self.min_snr_db)
            sgm = s * 10 ** (-snr_db / 20.)
            sample.add_(w.mul_(sgm))
        return sample


class RandomAdditiveVioletGN:
    def __init__(self, p=0.5, snr_db=35):
        self.snr_db = snr_db
        self.min_snr_db = 30
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            s = torch.sqrt(torch.mean(sample ** 2))
            n = sample.shape[-1]
            w = torch.randn(n)
            nn = n // 2 + 1
            k = torch.arange(1, nn + 1, 1).float()
            W = torch.fft.fft(w)
            W = W[:nn] * k
            W = torch.cat((W, W.flip(dims=(-1,))[1:-1].conj()), dim=-1)
            w = torch.fft.ifft(W).real
            w.add_(w.mean()).div_(w.std())
            snr_db = self.min_snr_db + torch.rand(1) * (self.snr_db - self.min_snr_db)
            sgm = s * 10 ** (-snr_db / 20.)
            sample.add_(w.mul_(sgm))
        return sample


class RandomAdditiveRedGN:
    def __init__(self, p=0.5, snr_db=35):
        self.snr_db = snr_db
        self.min_snr_db = 30
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            s = torch.sqrt(torch.mean(sample ** 2))
            n = sample.shape[-1]
            w = torch.randn(n)
            nn = n // 2 + 1
            k = torch.arange(1, nn + 1, 1).float()
            W = torch.fft.fft(w)
            W = W[:nn] / k
            W = torch.cat((W, W.flip(dims=(-1,))[1:-1].conj()), dim=-1)
            w = torch.fft.ifft(W).real
            w.add_(w.mean()).div_(w.std())
            snr_db = self.min_snr_db + torch.rand(1) * (self.snr_db - self.min_snr_db)
            sgm = s * 10 ** (-snr_db / 20.)
            sample.add_(w.mul_(sgm))
        return sample


class RandomAdditiveBlueGN:
    def __init__(self, p=0.5, snr_db=35):
        self.snr_db = snr_db
        self.min_snr_db = 30
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            s = torch.sqrt(torch.mean(sample ** 2))
            n = sample.shape[-1]
            w = torch.randn(n)
            nn = n // 2 + 1
            k = torch.arange(1, nn + 1, 1).float()
            W = torch.fft.fft(w)
            W = W[:nn] * k.sqrt()
            W = torch.cat((W, W.flip(dims=(-1,))[1:-1].conj()), dim=-1)
            w = torch.fft.ifft(W).real
            w.add_(w.mean()).div_(w.std())
            snr_db = self.min_snr_db + torch.rand(1) * (self.snr_db - self.min_snr_db)
            sgm = s * 10 ** (-snr_db / 20.)
            sample.add_(w.mul_(sgm))
        return sample


class RandomAddSine:
    def __init__(self, fs, snr_db=35, max_freq=50, p=0.5):
        self.snr_db = snr_db
        self.max_freq = max_freq
        self.min_snr_db = 30
        self.p = p
        self.fs = fs

    def __call__(self, sample):
        n = torch.arange(0, sample.shape[-1], 1)
        f = self.max_freq * torch.rand(1) + 3 * torch.randn(1)
        if random.random() < self.p:
            snr_db = self.min_snr_db + torch.rand(1) * (self.snr_db - self.min_snr_db)
            t = n * 1. / self.fs
            s = (sample ** 2).mean().sqrt()
            sgm = s * np.sqrt(2) * 10 ** (-snr_db / 20.)
            b = sgm * torch.sin(2 * np.pi * f * t + torch.rand(1) * np.pi)
            sample.add_(b)
        
        return sample


class RandomAmpSegment:
    def __init__(self, low, high, max_len=None, p=0.5):
        self.low = low
        self.high = high
        self.max_len = max_len
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            if self.max_len is None:
                self.max_len = sample.shape[-1] // 10
            idx = random.randint(0, self.max_len)
            amp = torch.FloatTensor(1).uniform_(self.low, self.high)
            sample[idx: idx + self.max_len].mul_(amp)
        return sample


class RandomCyclicShift:
    def __init__(self, max_time_shift=None, p=0.5):
        self.max_time_shift = max_time_shift
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            if self.max_time_shift is None:
                self.max_time_shift = sample.shape[-1]
            int_d = random.randint(0, self.max_time_shift - 1)
            if int_d > 0:
                sample = torch.cat((sample[-int_d:], sample[:-int_d]), dim=-1)
            else:
                pass
        return sample


class AudioAugs():
    def __init__(self, k_augs, fs, p=0.5, snr_db=30):
        self.noise_vec = ['awgn', 'abgn', 'apgn', 'argn', 'avgn', 'aun', 'phn', 'sine']
        augs = {}
        for aug in k_augs:
            if aug == 'amp':
                augs['amp'] = RandomAmp(p=p, low=0.5, high=1.3)
            elif aug == 'flip':
                augs['flip'] = RandomFlip(p)
            elif aug == 'neg':
                augs['neg'] = RandomAdd180Phase(p)
            elif aug == 'awgn':
                augs['awgn'] = RandomAdditiveWhiteGN(p=p, snr_db=snr_db)
            elif aug == 'abgn':
                augs['abgn'] = RandomAdditiveBlueGN(p=p, snr_db=snr_db)
            elif aug == 'argn':
                augs['argn'] = RandomAdditiveRedGN(p=p, snr_db=snr_db)
            elif aug == 'avgn':
                augs['avgn'] = RandomAdditiveVioletGN(p=p, snr_db=snr_db)
            elif aug == 'apgn':
                augs['apgn'] = RandomAdditivePinkGN(p=p, snr_db=snr_db)
            elif aug == 'mulaw':
                augs['mulaw'] = RandomMuLawCompression(p=p, n_channels=256)
            elif aug == 'tmask':
                augs['tmask'] = RandomTimeMasking(p=p, n_mask=int(0.1 * fs))
            elif aug == 'tshift':
                augs['tshift'] = RandomTimeShift(p=p, max_time_shift=int(0.1 * fs))
            elif aug == 'sine':
                augs['sine'] = RandomAddSine(p=p, fs=fs)
            elif aug == 'cycshift':
                augs['cycshift'] = RandomCyclicShift(p=p, max_time_shift=None)
            elif aug == 'ampsegment':
                augs['ampsegment'] = RandomAmpSegment(p=p, low=0.5, high=1.3, max_len=int(0.1 * fs))
            elif aug == 'aun':
                augs['aun'] = RandomAdditiveUN(p=p, snr_db=snr_db)            
            elif augs['timestretch']:
                augs['timestretch'] = RandomTimeStretch(p=p, fs=fs)
            else:
                raise ValueError("{} not supported".format(aug))
        self.augs = augs
        self.augs_signal = [a for a in augs if a not in self.noise_vec]
        self.augs_noise = [a for a in augs if a in self.noise_vec]

    def __call__(self, sample, **kwargs):
        augs = self.augs_signal.copy()
        augs_noise = self.augs_noise
        random.shuffle(augs)
        if len(augs_noise) > 0:
            i = random.randint(0, len(augs_noise) - 1)
            augs.append(augs_noise[i])
        for aug in augs:
            sample = self.augs[aug](sample)
        return sample


if __name__ == "__main__":
    pass