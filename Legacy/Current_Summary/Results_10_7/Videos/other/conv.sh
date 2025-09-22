for file in *.avi; do
    # Remove the .avi extension and assign the new filename with .mp4 extension
    output="${file%.avi}.mp4"

    # Run ffmpeg command for each .avi file
    ffmpeg -i "$file" -vcodec libx264 -acodec aac -strict -2 -pix_fmt yuv420p -movflags +faststart "$output"
done


