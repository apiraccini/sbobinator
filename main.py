from pathlib import Path
from utils import split_big_audio, transcribe_audio_chunks


raw_file_path = Path("data/raw") 
processed_files_path = Path("data/processed")
max_chunk_size_mb = 25

if not processed_files_path.exists():
    split_big_audio(raw_file_path, processed_files_path, max_chunk_size_mb)

transcripts_list = transcribe_audio_chunks(processed_files_path)
print(transcripts_list)