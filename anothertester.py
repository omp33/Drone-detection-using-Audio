from pydub import AudioSegment
import os

# Folder containing audio files
input_folder = r"C:\Users\Omprakash\Desktop\pthon\testinput"
output_folder = r"C:\Users\Omprakash\Desktop\pthon\testoutput"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Define trim duration (in milliseconds)
start_ms = 0       # start at 0 ms
end_ms = 6000      # trim to first 5 seconds

# Loop through files
for filename in os.listdir(input_folder):
    if filename.endswith((".wav", ".mp3")):  # adjust formats if needed
        file_path = os.path.join(input_folder, filename)
        
        # Load audio
        audio = AudioSegment.from_file(file_path)
        
        # Trim
        trimmed = audio[start_ms:end_ms]
        
        # Export trimmed file
        out_path = os.path.join(output_folder, filename)
        trimmed.export(out_path, format="wav")  # save as WAV
        print(f"Trimmed {filename} → {out_path}")
