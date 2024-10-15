import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, WhisperForConditionalGeneration
import ffmpeg
import os
from datetime import timedelta
from tqdm import tqdm
import re
import logging

# Use MPS (Metal Performance Shaders) for M2 Max
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32

# Change to a smaller model
model_id = "openai/whisper-large-v3"

model = WhisperForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    chunk_length_s=30,  # Process in 30-second chunks
    stride_length_s=5,   # 5-second overlap between chunks
    return_timestamps=True,
    generate_kwargs={"language": "ja", "task": "transcribe"}
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def extract_audio(video_path, audio_path):
    (
        ffmpeg
        .input(video_path)
        .output(audio_path, acodec='pcm_s16le', ac=1, ar='16k')
        .overwrite_output()
        .run(quiet=True)
    )

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def transcribe_audio(audio_path, pipe, output_srt):
    result = pipe(audio_path)
    
    with open(output_srt, 'w', encoding='utf-8') as srt_file:
        segment_id = 1
        
        # Create a progress bar based on the number of chunks
        pbar = tqdm(total=len(result["chunks"]), desc="Transcribing")
        
        for chunk in result["chunks"]:
            logger.debug(f"Processing chunk: {chunk}")
            start_time = chunk["timestamp"][0]
            end_time = chunk["timestamp"][1]
            text = chunk["text"].strip()
            logger.debug(f"Transcribed text: {text}")
            
            # Include segments with non-speech sounds or very short utterances
            if text or (end_time - start_time) > 0.5:  # Include if there's text or if the segment is longer than 0.5 seconds
                # Split text into sentences
                sentences = re.split('([。！？])', text)
                sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2])]
                
                lines = []
                current_line = ""
                for sentence in sentences:
                    if len(current_line) + len(sentence) <= 30:
                        current_line += sentence
                    else:
                        if current_line:
                            lines.append(current_line)
                        if len(sentence) <= 30:
                            current_line = sentence
                        else:
                            # Split long sentences
                            for char in sentence:
                                if len(current_line) < 30:
                                    current_line += char
                                else:
                                    lines.append(current_line)
                                    current_line = char
                
                if current_line:
                    lines.append(current_line)
                
                if not lines:
                    lines = [text if text else "[non-speech sound]"]  # Use placeholder for non-speech sounds
                
                srt_segment = f"{segment_id}\n{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n"
                srt_segment += "\n".join(lines) + "\n\n"
                srt_file.write(srt_segment)
                segment_id += 1
            
            pbar.update(1)
        
        # Close the progress bar
        pbar.close()

    return output_srt

# Example usage
video_path = "/Users/bailey/Downloads/【AIで音声をテキストに変換】Whisperの使い方を解説！〜 Pythonを使って無料でSpeech-to-Textを動かそう 〜.mp4"
audio_path = "temp_audio.wav"

# Generate the output SRT filename
video_filename = os.path.basename(video_path)
output_srt = f"SUBS-{os.path.splitext(video_filename)[0]}.srt"

# Extract audio from video
extract_audio(video_path, audio_path)

# Run speech recognition on the extracted audio and create SRT file
srt_filename = transcribe_audio(audio_path, pipe, output_srt)
print(f"SRT file created: {srt_filename}")

# Clean up the temporary audio file
os.remove(audio_path)
