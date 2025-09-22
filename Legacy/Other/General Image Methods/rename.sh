for file in "$directory"/*_trimmed.mp4; do
    # Extract the base filename without the "_trimmed"
    new_name="${file/_trimmed/}"
    
    # Rename the file
    mv "$file" "$new_name"
    echo "Renamed $file to $new_name"
done