#!/bin/bash

# Set base directory
base_dir="/Users/henryschnieders/Documents/Research/My_work/Current_Summary/Results_9_30/Videos"

# Define subdirectories for AVI and MP4 videos
avi_dir="$base_dir/avi"
mp4_dir="$base_dir/mp4"

# Output directory for trimmed videos
output_dir="$base_dir/trimmed"
mkdir -p "$output_dir/avi"
mkdir -p "$output_dir/mp4"

# Function to trim video
trim_video() {
    input_file="$1"
    output_file="$2"

    # Use ffmpeg to trim the first second of the video
    ffmpeg -ss 00:00:01 -i "$input_file" -c copy "$output_file"
}

# Process AVI files
for avi_file in "$avi_dir"/*.avi; do
    if [[ -f "$avi_file" ]]; then
        filename=$(basename -- "$avi_file")
        output_file="$output_dir/avi/${filename%.avi}_trimmed.avi"
        echo "Trimming $avi_file and saving as $output_file"
        trim_video "$avi_file" "$output_file"
    fi
done

# Process MP4 files
for mp4_file in "$mp4_dir"/*.mp4; do
    if [[ -f "$mp4_file" ]]; then
        filename=$(basename -- "$mp4_file")
        output_file="$output_dir/mp4/${filename%.mp4}_trimmed.mp4"
        echo "Trimming $mp4_file and saving as $output_file"
        trim_video "$mp4_file" "$output_file"
    fi
done

echo "Trimming completed for all videos."
