import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Visualise mean brightness of video frames")
parser.add_argument("file", type=str, default = '/Users/henryschnieders/Documents/Research/My_Data/testagain_6_10_wholeface.npy',  help="Path to the .npy file containing video frames")

args = parser.parse_args()

fn = Path(args.file)  # Get the file path from command line argument




# -------------------------------------------------
# 1.  load the list of frames
# -------------------------------------------------
raw = np.load(fn, allow_pickle=True)            # object-array

img = raw[0]

plt.imshow(img)
plt.axis('off')
plt.show()

raw = raw[:]
frames = raw.tolist() if isinstance(raw, np.ndarray) else raw



# sanity-check: make sure every element really is an image
for idx, f in enumerate(frames):
    if not (isinstance(f, np.ndarray) and f.ndim == 3):
        raise ValueError(f"Item {idx} is not an H×W×C image")

# -------------------------------------------------
# 2-A.  QUICK way – compute the mean one frame at a time
# -------------------------------------------------
frame_mean = np.array([f.mean(dtype=np.float64) for f in frames])

# -------------------------------------------------
# 2-B.  (optional) STACK into a 4-D cube first
#        – only works if all frames share the same shape
# -------------------------------------------------
# H, W, C = frames[0].shape
# video   = np.stack(frames, axis=0)             # (T, H, W, C)
# frame_mean = video.mean(axis=(1, 2, 3), dtype=np.float64)

# -------------------------------------------------
# 3.  visualise
# -------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(frame_mean, lw=1)
plt.xlabel("Frame")
plt.ylabel("Mean RGB intensity")
plt.title("Whole-frame mean brightness")
plt.grid(True)
plt.tight_layout()
plt.show()
