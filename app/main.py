import whisper
import os

def find_file_path(start_directory, file_extension):
    for dirpath, dirnames, filenames in os.walk(start_directory):
        for filename in filenames:
            if filename.endswith(file_extension):
                return os.path.join(dirpath, filename)
    return None 


start_directory = '/Users/enricomelis/Documents/1 Projects/scribe-app'  
file_extension = '.mp3'  
file_path = find_file_path(start_directory, file_extension)

if file_path:
    print(f'Found file at {file_path}')
else:
    print('File not found.')

model = whisper.load_model("base")
result = model.transcribe(file_path, fp16=False) 
transribed_text = result["text"]
print(transribed_text)