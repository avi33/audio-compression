import numpy as np
from psola import vocode

def gen_curve(n_segments, mode="fsf"):
    MAX = 1.5
    MIN = 0.2
    CONST = 1.0
    rates = [0.0] * n_segments

    if mode == "constant":
        rates = [CONST] * n_segments
    elif mode == "fsf":# fast-slow-fast(0.5-2-0.5)
        split = int(n_segments / 3)
        for i in range(split): rates[i] = MAX
        for i in range(split,split*2): rates[i] = MIN
        for i in range(split*2, n_segments): rates[i] = MAX
    elif mode == "parabola":
        x = np.array(range(n_segments))
        a = 4 * (MIN - MAX) / (n_segments * n_segments)
        rates = a * (x - n_segments / 2)**2 + MAX
    elif mode == "down":
        x = np.array(range(n_segments))
        rates = (MIN - MAX) / n_segments * x + MAX
    elif mode == "up":
        x = np.array(range(n_segments))
        rates = (MAX - MIN) / n_segments * x + MIN
    elif mode == "question":
        k = 4 * (MAX - 1) / n_segments
        for x in range(int(n_segments*0.75), n_segments): 
            rates[x] = max(1.0, k*x - 3*MAX + 4)
    elif mode == "stress":
        k = 4 * (1 - MAX) / n_segments
        for x in range(int(n_segments*0.5), int(n_segments*0.75)): 
            rates[x] =  k*x + 3*MAX - 2
    else:
        raise NotImplementedError   
    return rates

def change_rhythm_from_curve(audio, sr, mode="up", seg_size=0.16, silent_front=0.48, silent_end=0.32):
    seg_size = int(seg_size * sr)
    silent_front = int(silent_front / seg_size)
    silent_end = int(silent_end / seg_size)
    N = len(audio)

    if N % seg_size != 0:
        padding = int((N // seg_size + 1) * seg_size - N)
        audio = np.append(audio, [0.0]*padding)
        N = len(audio)
    assert(N % seg_size == 0)
    n_segments = int(N // seg_size - silent_front - silent_end)
    
    rates = [1.0] * silent_front + list(gen_curve(n_segments, mode)) + [1.0] * silent_end

    output_audio = []
    for i in range(n_segments):
        segment = audio[i*seg_size: (i+1)*seg_size]
        output_audio.append(vocode(audio=segment, sample_rate=sr, constant_stretch=rates[i]))

    output_audio = np.hstack(output_audio)
    
    return output_audio