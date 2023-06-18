#!/bin/bash

# Change directory to the workspace
cd ../workspace

# Run Whisper transcription on every .mp3 file in the current directory
for file in *.mp3
do
  # Get the base name of the file (i.e., without the .mp3 extension)
  base="${file%.*}"
  
  # Resample to 16kHz using ffmpeg
  ffmpeg -y -i "$file" -ar 16000 -ac 1 -c:a pcm_s16le "${base}_16.wav"
  
  # Transcribe the resampled file using Whisper
  whisper "${base}_16.wav"

  # Export the transcription to a .txt file
  mv "${base}_16.wav.txt" "${base}.txt"

  # Export the transcription with timestamps to a .ts.txt file
  # This assumes that Whisper can export timestamps, which might not be the case
  # If not, you may need to use a different tool for this part
  # mv "${base}_16.wav.vtt" "${base}.ts.txt" 
done
