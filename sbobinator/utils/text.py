from pathlib import Path
import os
import pickle
import tiktoken
from tqdm import tqdm 

import markdown2
import pdfkit

from openai import OpenAI
from dotenv import load_dotenv


system_prompt = """
You are a helpful assistant. You will receive a raw transcription from a call to the Whisper API. I need you to to process the text doing the following tasks:
- fix and add punctuation, if necessary
- fix typos and general grammar or syntax errors
- organize the text into sections, subsections (each with titles) and paragraphs
This is very important, take a big breath and do this carefully. Use markdown syntax for headers and subheaders
"""

def prepare_messages(transcript, system_prompt):

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f'Raw transcript:\n\n<<<{transcript}>>>\n\n'}   
    ]

    return messages


def num_tokens_from_messages(messages, model="gpt-4-0613"):
    """Return the number of tokens used by a list of messages."""
    
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    
    return num_tokens


def process_text(transcripts_list, output_folder, model):

    load_dotenv()
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    responses = []
    for transcript in tqdm(transcripts_list):
        response = client.chat.completions.create(
            model=model,
            messages=prepare_messages(transcript, system_prompt),
        )
        text = response.choices[0].message.content
        responses.append(text)
    out = ' '.join(responses)

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_path / "processed_text.pkl"
    with open(output_file_path, 'wb') as file:
        pickle.dump(out, file)
    
    return out

def text2pdf(text):

    html_content = markdown2.markdown(text)
    output_path = Path("data/out")
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / 'transcript.html', 'w', encoding='utf-8') as file:
        file.write(html_content)

    pdfkit.from_file(output_path / 'transcript.html', output_path / 'transcript.pdf')


if __name__ == '__main__':

    final_data_path = Path("data/final")
    transcripts_path = final_data_path / "transcripts.pkl"

    with open(transcripts_path, 'rb') as file:
        transcripts_list = pickle.load(file)
    transcripts = ' '.join(transcripts_list)

    messages = prepare_messages(transcripts, system_prompt)
    total_tokens = num_tokens_from_messages(messages)
    
    print(total_tokens)
    print(messages)