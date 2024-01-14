from pathlib import Path
import pickle
from utils.audio import split_big_audio, transcribe_audio_chunks
from utils.text import process_text, text2html


def main():
    
    # constants and paths
    raw_audio_format = 'm4a'
    audio_language = 'it'

    raw_file_path = Path("data/raw") 
    max_chunk_size_mb = 25
    processed_files_path = Path("data/processed")
    final_data_path = Path("data/final")
    output_path = Path("data/out")
    transcripts_path = final_data_path / "transcripts.pkl"

    print('\nI am Sbobinator. I will help you with your transcription with the power of AI.\n\n')

    # Step 1: split audio files if not processed
    if not processed_files_path.exists():
        split_big_audio(input_folder=raw_file_path, output_folder=processed_files_path, max_chunk_size_mb=max_chunk_size_mb)
        print("Audio files split into smaller chunks.")

    # Step 2: transcribe audio chunks if transcripts file doesn't exist
    if not transcripts_path.exists():
        transcribe_audio_chunks(audio_chunks_path=processed_files_path, output_folder=final_data_path)
        print("Audio chunks transcribed.")

    # Step 4: generate HTML from raw transcript text
    with open(transcripts_path, 'rb') as file:
        text = ' '.join(pickle.load(file))
    text2html(text, name='raw_transcript')
    print("HTML generated from raw unprocessed text.")

    # Step 4: process text if processed text file doesn't exist
    processed_text_path = final_data_path / "processed_text.pkl"
    if not processed_text_path.exists():
        with open(transcripts_path, 'rb') as file:
            transcripts_list = pickle.load(file)
        processed_text = process_text(transcripts_list, final_data_path, model='gpt-3.5-turbo-16k-0613')
        print("Text processed.")
    
    # Step 5: generate HTML from processed text
    if not Path(output_path / "transcript.html").exists():
        with open(processed_text_path, 'rb') as file:
            text = pickle.load(file, name='transcript')
        text2html(text)
        print("HTML generated from processed text.")

if __name__ == "__main__":
    main()
