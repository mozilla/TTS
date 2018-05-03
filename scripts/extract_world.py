import os
import sys
sys.path.append("/home/erogol/projects/")
import time
import glob
import soundfile as sf
import numpy as np
import tqdm
import pyworld as pw

from multiprocessing import Pool

DATA_PATH = "/data/shared/KeithIto/LJSpeech-1.0/"
OUT_PATH = "/data/shared/KeithIto/LJSpeech-1.0/world2/"
FFT_SIZE = 1024

def world_decode(x, fs):
    _f0_h, t_h = pw.harvest(x, fs)
    f0_h = pw.stonemask(x, _f0_h, t_h, fs, fft_size=FFT_SIZE)
    sp_h = pw.cheaptrick(x, f0_h, t_h, fs, fft_size=FFT_SIZE)
    ap_h = pw.d4c(x, f0_h, t_h, fs)
    return f0_h, sp_h, ap_h

    
def world_encode(f0_h, sp_h, ap_h, fs):
    y_h_harvest = pw.synthesize(f0_h, sp_h, ap_h, fs, pw.default_frame_period)
    

def extract_world(file_path):
    x, fs = sf.read(file_path)
    f0_h, sp_h, ap_h = world_decode(x, fs)
    file_name = os.path.basename(file_path).replace(".wav","")
    f0_file = file_name + ".f0"
    sp_file = file_name + ".sp"
    ap_file = file_name + ".ap"
    np.save(os.path.join(OUT_PATH, f0_file), f0_h, allow_pickle=False)
    np.save(os.path.join(OUT_PATH, sp_file), sp_h, allow_pickle=False)
    np.save(os.path.join(OUT_PATH, ap_file), ap_h, allow_pickle=False)

glob_path = os.path.join(DATA_PATH, "**/*.wav")
file_names = glob.glob(glob_path, recursive=True)

if __name__ == "__main__":
    print(" > Number of files: %i"%(len(file_names)))
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
        print(" > A new folder created at {}".format(OUT_PATH))

    with Pool(20) as p:
        r = list(tqdm.tqdm(p.imap(extract_world, file_names), total=len(file_names)))