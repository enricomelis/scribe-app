import whisper
import os

def find_file_path(start_directory, file_extension):
    for dirpath, dirnames, filenames in os.walk(start_directory):
        for filename in filenames:
            if filename.endswith(file_extension):
                return os.path.join(dirpath, filename)
    return None 


import whisper

def transcribe_audio(filename):
    try:
        with open(filename, 'rb') as f:
            print(f'Found file at {filename}')
    except FileNotFoundError:
        print('File not found.')
        return
    
    model = whisper.load_model("medium")
    result = model.transcribe(filename, fp16=False) 
    transcribed_text = result["text"]
    
    return transcribed_text




