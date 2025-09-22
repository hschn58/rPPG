import subprocess
import time
import os

# Set recording parameters
duration = 20  # seconds
fps = 30       # Supported framerate
width = 1760   # Supported resolution
height = 1328

# Prompt for filename
description = input("Enter the description of the video: ")
initial_filename = input("Enter the initial filename (without extension, e.g., 'my_video'): ")
output_file = f'/Users/henryschnieders/Documents/Research/My_work/Current_Summary/Starting_4_5_2025/Data/{initial_filename}.mp4'

# FFmpeg command for 10-bit H.264 recording
ffmpeg_command = [
    'ffmpeg',
    '-f', 'avfoundation',
    '-i', '"FaceTime HD Camera"',  # MacBook’s built-in camera
    '-pix_fmt', 'yuv420p10le',     # 10-bit pixel format
    '-c:v', 'libx264',             # H.264 codec
    '-profile:v', 'high10',        # 10-bit profile
    '-r', str(fps),                # Frame rate
    '-s', f'{width}x{height}',     # Resolution
    '-t', str(duration),           # Duration
    output_file                    # Output file
]

# Start FFmpeg recording
print("Starting 10-bit recording...")
process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

# Check for errors
if process.returncode == 0:
    print(f"Video saved as {output_file}")
else:
    print("Error during recording:")
    print(stderr.decode())

# Optional: Rename file
rename_choice = input("Would you like to rename the file? (yes/no): ").lower()
if rename_choice == 'yes':
    new_filename = input("Enter the new filename (without extension): ")
    new_output_file = f'/Users/henryschnieders/Documents/Research/My_work/Current_Summary/Starting_4_5_2025/Data/{new_filename}.mp4'
    os.rename(output_file, new_output_file)
    print(f"Video renamed to {new_output_file}")
else:
    print("File not renamed.")