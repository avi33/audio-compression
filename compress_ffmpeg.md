"amr"
supported bitrates - 4.75k, 5.15k, 5.9k, 6.7k, 7.4k, 7.95k, 10.2k or 12.2k
ffmpeg -i arctic_a0001.wav -ar 8000 -ab 12.2k arctic_a0001.amr


"aac"
supported bitrates - 32k, 64k, 96k
ffmpeg -i arctic_a0001.wav -codec:a aac -ab 32k arctic_a0001.aac
