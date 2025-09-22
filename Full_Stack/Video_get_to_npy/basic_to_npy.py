import cv2
import numpy as np
import subprocess
import os
import pickle
import time
import re

import objc
import AVFoundation as AVF

from Cocoa import NSObject
import AVFoundation as AVF, time

import csv

import argparse

def get_params_from_cli_or_prompt():
    """
    • First try to grab --foo and --bar from the command line.
    • If the user didn't provide them, fall back to asking interactively.
    """
    p = argparse.ArgumentParser(description="get video acquisition save namee")
    p.add_argument("--video_name", type = str, help = 'pixel trajectories file path' )               # default is None
    args = p.parse_args()

    if args.video_name is None:
        args.video_name = input('Enter video save name (without extension): ')

    return args.video_name


video_name = get_params_from_cli_or_prompt()  # Get video name from command line or prompt

_lock_session = None       # <- keep a strong ref so GC won't close it

"""
This code uses the template matching method. First, the face is found using the YuNet face detection model. 
Next, the face is tracked using template matching on the grayscale image, updated every frame. Five regions of the 
face are analyzed in each frame: the chin, midleft and midright cheek, right and left cheek. 

Specs of the output video:

    video quality:    10-bit
    duration:         10 seconds
    fps:              30
    width:            1760 pixels
    height.           1328 pixels

In each saved frame:

    region1: chin
    region2: left cheek
    region3: right cheek
    region4: midleft cheek
    region5: midright cheek


# Recommended FFmpeg command for 10-bit capture on macOS
ffmpeg_command = [
    "ffmpeg",
    "-f", "avfoundation",red_p4
    10
    
    "-framerate", str(fps),
    "-pixel_format", "420v",          # what most webcams really output
    "-i", "0",

    # ----- colour-space / bit-depth handling -----
    # If your camera is *already* 10-bit (rare), drop the -vf line.
    "-vf", "format=yuv420p10le",      # promote 8-bit input to 10-bit

    # ----- encoder -----
    "-c:v", "hevc_videotoolbox",      # hardware HEVC encoder, 10-bit capable
    "-profile:v", "main10",
    "-pix_fmt", "yuv420p10le",

    # ----- real-time safeguards -----
    "-preset", "fast",                # (optional) a little extra head-room
    "-vsync", "2", "-drop_end", "1",  # drop late frames instead of freezing

    # ----- container settings -----
    "-r", str(fps),
    "-s", f"{width}x{height}",
    "-t", str(duration),
    "-y",                             # overwrite existing file
    output_video
]
"""
save_path =  "/Users/henryschnieders/Documents/Research/My_Data"


# ──── 1.  add just after your imports ───────────────────────────────────────────


def max_fps_for_avfoundation(dev_index: str, width: int, height: int, fallback: int = 30) -> int:
    """
    Query FFmpeg for the highest fps the selected AVFoundation device supports
    at the requested resolution.  Returns an int (rounded) or `fallback`.
    """
    try:
        probe = subprocess.run(
            [
                "ffmpeg", "-hide_banner",
                "-f", "avfoundation", "-list_formats", "all", "-i", dev_index
            ],
            capture_output=True, text=True, check=True
        ).stderr  # list is printed on stderr
        pattern = rf"{width}x{height}.*?(\d+(?:\.\d+)?)\s*fps"
        fps_values = [float(m.group(1)) for m in re.finditer(pattern, probe)]
        return int(round(max(fps_values))) if fps_values else fallback
    except Exception:
        return fallback
# ────────────────────────────────────────────────────────────────────────────────

def preview_and_lock_camera(seconds=1.0, device_index=0):
    """
    Let AE / AWB converge, then lock them.  Keeps the session alive so FFmpeg
    records with the same manual settings.  Autofocus is left ON.
    """
    global _lock_session

    # 0️⃣   list devices so you know which index to choose
    video_devs = AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo)
    if not video_devs:
        print("❌  No video devices found – skipping lock.")
        return
    print("-- Available cameras --")
    for idx, d in enumerate(video_devs):
        print(f"  [{idx}] {d.localizedName()}")

    if device_index >= len(video_devs):
        print(f"⚠️  Index {device_index} out of range – skipping lock.")
        return
    dev = video_devs[device_index]

    # 1️⃣   ensure the app has permission
    status = AVF.AVCaptureDevice.authorizationStatusForMediaType_(AVF.AVMediaTypeVideo)
    if status == AVF.AVAuthorizationStatusNotDetermined:
        ok = AVF.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AVF.AVMediaTypeVideo,  # modal 1-shot popup
            lambda granted: None
        )
        # requestAccess… is async; give user a moment to click “OK”
        for _ in range(20):
            time.sleep(0.1)
            if AVF.AVCaptureDevice.authorizationStatusForMediaType_(AVF.AVMediaTypeVideo) \
                    != AVF.AVAuthorizationStatusNotDetermined:
                break

    status = AVF.AVCaptureDevice.authorizationStatusForMediaType_(AVF.AVMediaTypeVideo)
    if status != AVF.AVAuthorizationStatusAuthorized:
        print("❌  Camera access denied – skipping lock.")
        return

    # 2️⃣   start a throw-away session
    ses = AVF.AVCaptureSession.alloc().init()
    inp, err = AVF.AVCaptureDeviceInput.deviceInputWithDevice_error_(dev, None)
    if inp is None:
        print(f"❌  Could not open device: {err.localizedDescription() if err else 'Unknown'}")
        return

    ses.beginConfiguration()
    ses.addInput_(inp)
    ses.commitConfiguration()
    ses.startRunning()

    time.sleep(seconds)            # wait for AE/AWB to settle

    # 3️⃣   lock exposure & white-balance, leave AF alone
    if dev.lockForConfiguration_(None):
        if dev.isExposureModeSupported_(AVF.AVCaptureExposureModeLocked):
            dev.setExposureMode_(AVF.AVCaptureExposureModeLocked)
        if dev.isWhiteBalanceModeSupported_(AVF.AVCaptureWhiteBalanceModeLocked):
            dev.setWhiteBalanceMode_(AVF.AVCaptureWhiteBalanceModeLocked)
        dev.unlockForConfiguration()
        print("✅  AE and WB locked; AF still active.")
    else:
        print("⚠️  Could not lock device for configuration – skipping lock.")

    _lock_session = ses          # keep session alive

    if dev.isExposureModeSupported_(AVF.AVCaptureExposureModeCustom):
        dur  = dev.exposureDuration()      # current settled values
        iso  = dev.ISO()
        dev.setExposureModeCustomWithDuration_ISO_completionHandler_(dur, iso, None)

    if dev.lockForConfiguration_(None):
        if dev.isExposureModeSupported_(AVF.AVCaptureExposureModeLocked):
            dev.setExposureMode_(AVF.AVCaptureExposureModeLocked)
        if dev.isWhiteBalanceModeSupported_(AVF.AVCaptureWhiteBalanceModeLocked):
            dev.setWhiteBalanceMode_(AVF.AVCaptureWhiteBalanceModeLocked)
        if dev.isSubjectAreaChangeMonitoringEnabled():
            dev.setSubjectAreaChangeMonitoringEnabled_(False)   # <- add
        dev.unlockForConfiguration()


# Process frames with template matching
def process_frames(frames, initial_hr, final_hr, out):
    
    frames_npy = []
    
    for i, frame in enumerate(frames):
        # Flip frame vertically (as requested)
        #frame = cv2.flip(frame, 1)
        
        frames_npy.append(frame)
    
        # Write flipped frame to video
        frame = cv2.flip(frame, 1)  # Flip frame back to original orientation
                # Overlay heart rate
        if i == 0:
            cv2.putText(frame, f"Initial HR: {initial_hr} bpm", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        elif i == len(frames) - 1:
            cv2.putText(frame, f"Final HR: {final_hr} bpm", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        out.write(frame)
    
    return frames_npy

# Main script

# Recording parameters

cut_param = 4
duration  = 12 + cut_param  
width  = None   # let FFmpeg tell us
height = None
base_path = save_path  # "/Users/henryschnieders/Documents/Research/My_Data"

fps = max_fps_for_avfoundation("0", width, height)   # auto-detect
print(f"Using {fps} fps (maximum supported for {width}×{height})")

# User inputs

initial_hr = int(input("Enter initial heart rate (bpm): "))

# 1) Write raw FFmpeg output to a “_raw.mp4” file
raw_video      = f"{base_path}/{video_name}_raw.mp4"
processed_video = f"{base_path}/{video_name}_processed.mp4"
output_data    = f"{base_path}/{video_name}_wholeface.npy"

ffmpeg_command = [
    "ffmpeg",
    "-f", "avfoundation",
    "-framerate", str(fps),
    "-pixel_format", "nv12",
    "-i", "0",
    "-c:v", "h264_videotoolbox",
    "-profile:v", "high",
    "-pix_fmt", "yuv420p",
    "-preset", "fast",
    "-fps_mode", "cfr",
    "-r", str(fps),
    "-t", str(duration),
    "-y",
    raw_video,                          # <— write to “raw_video”, not directly to “video_name.mp4”
]

#lock AE, AWB settings after 1 second into video

preview_and_lock_camera(seconds=1.0, device_index=0)


#start recording 
print("Recording video for 10 seconds...")
proc = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


stdout, stderr = proc.communicate()
if proc.returncode != 0:
    print("Error during recording:")
    print(stderr.decode())
    exit(1)

# ─── 2) Load the “raw_video” for processing ╌───────────────────────────────────
cap = cv2.VideoCapture(raw_video)
frames = []
count = 0

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count < cut_param*0.75 * fps or count > (duration - cut_param*0.25) * fps:
        continue

    frames.append(frame)

cap.release()


# Check gain coefficients for AWB
# ──── AWB-gain extraction & CSV save ───────────────────────────────────────────
# Compute per-frame channel means
R_mean, G_mean, B_mean = [], [], []
for f in frames:                            # each f is (H,W,3)   BGR from OpenCV
    B, G, R = cv2.split(f.astype(np.float32))
    R_mean.append(R.mean())
    G_mean.append(G.mean())
    B_mean.append(B.mean())

R_mean = np.asarray(R_mean)
G_mean = np.asarray(G_mean)
B_mean = np.asarray(B_mean)

# Gain (channel-ratio) traces
eps = 1e-6                                  # avoid /0
RG = R_mean / (G_mean + eps)
BG = B_mean / (G_mean + eps)

# Save to CSV
gain_csv = f"{base_path}/{video_name}_awb_gains.csv"
with open(gain_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "R_mean", "G_mean", "B_mean", "R_over_G", "B_over_G"])
    for i, (r, g, b, rg, bg) in enumerate(zip(R_mean, G_mean, B_mean, RG, BG)):
        w.writerow([i, r, g, b, rg, bg])

print(f"✓  AWB gain coefficients saved to {gain_csv}")


# ─── 3) Set up VideoWriter to create only the “_processed.mp4” ╌──────────────────
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

height, width = frames[0].shape[:2]        # <- real, physical size
print(f"Native capture size: {width}×{height}")

out = cv2.VideoWriter(
    processed_video,  # <— write only to “_processed.mp4”
    fourcc,
    fps,
    (width, height)
)

if not frames:
    raise RuntimeError("No frames captured!")


final_hr = int(input("Enter final heart rate (bpm): "))

print("Processing frames with template matching…")
all_regions = process_frames(frames, initial_hr, final_hr, out)

# ─── 4) Save your region data and clean up ╌────────────────────────────────────
with open(output_data, 'wb') as f:
    pickle.dump(all_regions, f)
print(f"Saved region data to {output_data}")

out.release()
cv2.destroyAllWindows()

# ─── 5) Delete the raw FFmpeg file so only the processed file remains ───────────
try:
    os.remove(raw_video)
    print(f"Deleted temporary raw video: {raw_video}")
except OSError:
    print(f"Could not delete raw file: {raw_video}")

print(f"Processed video saved as {processed_video}")