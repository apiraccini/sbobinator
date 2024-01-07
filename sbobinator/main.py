from pathlib import Path
import pickle
from utils.audio import split_big_audio, transcribe_audio_chunks
from utils.text import process_text, text2pdf


raw_file_path = Path("data/raw") 
max_chunk_size_mb = 25
processed_files_path = Path("data/processed")
final_data_path = Path("data/final")
transcripts_path = final_data_path / "transcripts.pkl"


if not processed_files_path.exists():
    split_big_audio(raw_file_path, processed_files_path, max_chunk_size_mb)

if not transcripts_path.exists():
    transcribe_audio_chunks(processed_files_path, final_data_path)

with open(transcripts_path, 'rb') as file:
    transcripts_list = pickle.load(file)

text = process_text(transcripts_list, final_data_path, model='gpt-4-0613')
with open(final_data_path / "processed_text.pkl", 'rb') as file:
    text = pickle.load(file)

print(text)
text2pdf(text)