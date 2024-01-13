from pathlib import Path
import pickle
from utils.audio import split_big_audio, transcribe_audio_chunks
from utils.text import process_text, text2html


def main():
    
    # constants and paths
    raw_file_path = Path("data/raw") 
    max_chunk_size_mb = 25
    processed_files_path = Path("data/processed")
    final_data_path = Path("data/final")
    transcripts_path = final_data_path / "transcripts.pkl"

    print('\nThis is Sbobinator. I will help you with your transcription with the power of AI.\n\n')

    # Step 1: split audio files if not processed
    if not processed_files_path.exists():
        split_big_audio(raw_file_path, processed_files_path, max_chunk_size_mb)
        print("Audio files split into smaller chunks.")

    # Step 2: transcribe audio chunks if transcripts file doesn't exist
    if not transcripts_path.exists():
        transcribe_audio_chunks(processed_files_path, final_data_path)
        print("Audio chunks transcribed.")

    # Step 3: process text if processed text file doesn't exist
    processed_text_path = final_data_path / "processed_text.pkl"
    if not processed_text_path.exists():
        with open(transcripts_path, 'rb') as file:
            transcripts_list = pickle.load(file)
        processed_text = process_text(transcripts_list, final_data_path, model='gpt-3.5-turbo-16k-0613')
        print("Text processed.")

    # # Step 4: generate HTML from raw transcript text
    # if not Path(final_data_path / "raw_transcript.html").exists():
    #     with open(processed_text_path, 'rb') as file:
    #         text = pickle.load(file)
    #     text2html(text)
    #     print("HTML generated from processed text.")

    # Step 5: generate HTML from processed text
    if not Path(final_data_path / "transcript.html").exists():
        with open(processed_text_path, 'rb') as file:
            text = pickle.load(file)
        text2html(text)
        print("HTML generated from processed text.")

if __name__ == "__main__":
    main()
