from pathlib import Path
import os
import pickle
from tqdm import tqdm

from pydub import AudioSegment

from openai import OpenAI
from dotenv import load_dotenv


def split_big_audio(input_folder, output_folder, max_chunk_size_mb=20):
    """
    Splits a single large MP3 file into smaller chunks with a maximum size specified in megabytes
    while ensuring each split is also adjusted based on the total duration of the audio file.

    Parameters:
    - input_file (str): Path to the input MP3 file.
    - output_folder (str): Path to the folder where the split MP3 files will be saved.
    - max_chunk_size_mb (int): Maximum size of each chunk in megabytes. Default is 20 MB.

    The function loads the MP3 file, determines its total size,
    calculates the number of chunks based on the maximum chunk size,
    and creates the smaller chunks, saving them in the specified output folder.
    """

    input_path = Path(input_folder)
    files = list(input_path.glob("*.mp3"))
    
    if len(files) != 1:
        print("Please provide a folder containing exactly one MP3 file.")
        return

    size_in_bytes = Path(files[0]).stat().st_size
    size_in_megabytes = size_in_bytes / (1024 * 1024)
    print(f"The size of the loaded MP3 file is approximately: {size_in_megabytes:.2f} Mb.")
    
    max_chunk_size_bytes = max_chunk_size_mb * 1024 * 1024
    total_chunks = (size_in_bytes + max_chunk_size_bytes - 1) // max_chunk_size_bytes

    large_mp3 = AudioSegment.from_file(files[0], format="mp3")

    total_duration_ms = len(large_mp3)
    max_chunk_duration_ms = total_duration_ms // total_chunks
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for chunk_number, start in tqdm(enumerate(range(0, len(large_mp3), max_chunk_duration_ms), start=1), total=total_chunks):
        end = start + max_chunk_duration_ms if start + max_chunk_duration_ms < len(large_mp3) else len(large_mp3)
        chunk_segment = large_mp3[start:end]
        output_path_chunk = output_path / f"split_{chunk_number}.mp3"
        chunk_segment.export(output_path_chunk, format="mp3")
    
    print('Done.')


def transcribe_audio_chunks(audio_chunks_path, output_folder):
    """
    Transcribes each audio chunk in the specified directory using the OpenAI API 
    and saves the list of trascriptions in a pickle file.

    Parameters:
    - audio_chunks_path (Path): Pathlib Path object pointing to the directory containing audio chunks.
    """
    
    load_dotenv()
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    transcripts = []
    for audio_file in tqdm(audio_chunks_path.glob("*.mp3")):
        with open(audio_file, 'rb') as audio:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio, 
                response_format="text"
                )
            transcripts.append(transcript)

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_path / "transcripts.pkl"
    with open(output_file_path, 'wb') as file:
        pickle.dump(transcripts, file)

if __name__ == '__main__':
    
    raw_file_path = Path("data/raw") 
    processed_files_path = Path("data/processed")
    max_chunk_size_mb = 20 

    split_big_audio(raw_file_path, processed_files_path, max_chunk_size_mb)
