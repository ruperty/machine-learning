# read mp3 files from the audio directoty and list them

import os

def read_audio_file_from_list():
    audio_files = []
    for root, dirs, files in os.walk("audio"):
        for file in files:
            if file.endswith(".mp3"):
                audio_files.append(file)
    return audio_files
    
print(read_audio_file_from_list())

