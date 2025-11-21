# Import necessary libraries
from transformers import pipeline, AutoTokenizer
from huggingface_hub import login
import torch
from gtts import gTTS
import soundfile as sf
import pexels
import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip, concatenate_audioclips
from subtitles import VideoTranscriber
import os
import moviepy.video.fx.all as vfx
from pytube import YouTube
import youtube
import time
from urllib.error import URLError, HTTPError
from pydub import AudioSegment
import re
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QApplication, QPushButton, QLabel, QLineEdit, QVBoxLayout, QWidget

# Login to Hugging Face Hub
login(token="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")

# Global variables
music_keywords = None
duration_seconds = None
keywords_result = None

# Helper functions
def with_opencv(filename):
    """Calculate the length of a video using OpenCV."""
    video = cv2.VideoCapture(filename)
    if not video.isOpened():
        print(f"Error opening video file: {filename}")
        return 0
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    video.release()
    return frame_count / fps

def sanitize_filename(filename):
    """Remove special characters and limit the length of the filename."""
    sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
    return sanitized[:100]  # Limit filename length to 100 characters

def download_youtube_audio(video_url, output_file, max_retries=3):
    """Download audio from a YouTube video."""
    for attempt in range(max_retries):
        try:
            yt = YouTube(video_url)
            audio_stream = yt.streams.filter(only_audio=True).first()
            sanitized_title = sanitize_filename(yt.title)
            output_file = f"{sanitized_title}.mp4"
            audio_stream.download(filename=output_file)
            return output_file
        except (URLError, HTTPError) as e:
            print(f"Download failed: {e}. Retrying ({attempt + 1}/{max_retries})...")
            time.sleep(2 ** attempt)
    print(f"Failed to download audio after {max_retries} attempts.")
    return None

def extract_audio(video_file, audio_file):
    """Extract audio from a video file and save it as an audio file."""
    audio = AudioSegment.from_file(video_file, "mp4")
    audio.export(audio_file, format="wav")
    return audio_file

def download_background_music(keywords, target_duration):
    """Download background music from YouTube based on keywords and target duration."""
    results = youtube.youtube_search(keywords)
    print("YouTube search results:", results)  # Debug print
    
    music_clips = []
    total_music_length = 0

    for video in results:
        url = video['url']
        title = video['title']
        file_name = f"{title}.mp4"
        
        downloaded_file = download_youtube_audio(url, file_name)
        if downloaded_file:
            print(f"Downloaded file: {downloaded_file}")  # Debug print
            # Extract audio from the downloaded video file
            audio_file = extract_audio(downloaded_file, f"{title}.wav")
            audio_info = sf.info(audio_file)
            music_length = audio_info.duration
            total_music_length += music_length
            print(f"Extracted audio file: {audio_file}, length: {music_length} seconds")  # Debug print
            
            music_clip = AudioFileClip(audio_file)
            music_clips.append(music_clip)
            
            if total_music_length >= target_duration:
                break

    if music_clips:
        combined_music = concatenate_audioclips(music_clips)
    else:
        combined_music = None

    return combined_music


def videoDownloader(keywords, audio_len):
    """Download videos from Pexels to match the audio length."""
    pexels_api = pexels.PexelsAPI("H1Cg4ewyLgmaYkpZ2TZFOAiMzDUX4qo1Bqr4I6dedjjuef0CuKUzeG3V")
    try:
        videos_1 = pexels_api.searchVideo(query=keywords[0], per_page=10)
        videos_2 = pexels_api.searchVideo(query=keywords[1], per_page=10)
    except:
        print("Unable to search for video")

    # Print the response to understand its structure
    print("Response for videos_1:", videos_1)
    print("Response for videos_2:", videos_2)

    # Initialize variables
    i = 0
    video_len = 0
    video_files = []

    # Check if videos are found
    if not videos_1 and not videos_2:
        print("No videos found for the given keywords.")
        return

    for video_url in videos_1:
        video_file = f"video{i}.mp4"
        pexels_api.downloadVideo(video_url, video_file)
        video_duration = with_opencv(video_file)
        print(f"Video {i} duration: {video_duration} seconds")
        video_len += video_duration
        video_files.append(video_file)
        if video_len >= audio_len:
            break
        i += 1

    if video_len < audio_len:
        for video_url in videos_2:
            video_file = f"video{i}.mp4"
            pexels_api.downloadVideo(video_url, video_file)
            video_duration = with_opencv(video_file)
            print(f"Video {i} duration: {video_duration} seconds")
            video_len += video_duration
            video_files.append(video_file)
            if video_len >= audio_len:
                break
            i += 1

    # Download background music
    background_music = download_background_music(music_keywords, audio_len)
    
    # Combine clips with background music
    combine_clips(video_files, audio_len, background_music)
    
    # Clean up downloaded video files
    for j in range(i + 1):
        os.remove(f"video{j}.mp4")

def combine_clips(clips, target_duration, background_music):
    """Combine video clips and add background music."""
    footages = []
    total_duration = 0
    target_width, target_height = 1080, 1920

    for clip in clips:
        footage = VideoFileClip(clip)
        original_width, original_height = footage.size
        aspect_ratio = original_width / original_height
        target_aspect_ratio = target_width / target_height

        if (aspect_ratio > target_aspect_ratio):
            new_height = target_height
            new_width = new_height * aspect_ratio
        else:
            new_width = target_width
            new_height = new_width / aspect_ratio

        resized_footage = footage.resize(newsize=(new_width, new_height))
        cropped_footage = vfx.crop(
            resized_footage,
            width=target_width,
            height=target_height,
            x_center=new_width // 2,
            y_center=new_height // 2
        )

        if (total_duration + cropped_footage.duration > target_duration):
            cropped_footage = cropped_footage.subclip(0, target_duration - total_duration)
            footages.append(cropped_footage)
            total_duration += cropped_footage.duration
            break
        footages.append(cropped_footage)
        total_duration += cropped_footage.duration

    # Load the audio file
    audio_file = AudioFileClip("script.mp3")

    # Ensure the background music is trimmed to the target duration
    if background_music:
        print("Background music found, duration:", background_music.duration)  # Debug print
        if (background_music.duration > target_duration):
            background_music = background_music.subclip(0, target_duration)
        # Combine the audio tracks
        background_music = background_music.volumex(0.125)
        all_audio = CompositeAudioClip([audio_file, background_music.set_duration(target_duration)])
    else:
        print("No background music found.")  # Debug print
        all_audio = audio_file

    final_clip = concatenate_videoclips(footages, method="compose").set_audio(all_audio)
    final_clip.write_videofile("final.mp4", codec="libx264", temp_audiofile="temp-audio.m4a", remove_temp=True, audio_codec="aac")

    model_name = "base"
    video_path = "final.mp4"
    subtitles = VideoTranscriber(model_name, video_path, "Roboto-Black.ttf")
    subtitles.transcribe_video()
    subtitles.create_video("output_with_subtitles.mp4")


def Run():
    """Main function to run the video generation process."""
    global music_keywords, keywords_result, duration_seconds, topic

    topic = text_field.text()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16, device_map="auto")

    def generateAnswer(prompt):
        outputs = pipe(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=512
        )
        return outputs[0]['generated_text']

    query = (
        f"Create a 30-second minimum script for a YouTube short about {topic}. "
        "Include only the speaking part without any segments. Ensure the script loops seamlessly and do not include music recommendations. "
        "Label the end as [END]. Skip two lines after each sentence. "
        "Label the start with [START]."
        "After the [END] put 2 related keywords to the script and separate them using a comma, do not put a space after the comma"
        "At the start of the keywords place a [KEYWORDS] and at the end place [KEYWORDSEND]"
        "Start the video with an interesting question that will keep the viewers hooked"
        "Add some related music at the end and place it in between [MUSIC] and [MUSIC_END]. Make sure that the music is a genre of music. The keywords shall be separated by a comma"
        "Include an outro"
    )

    # Generate the script
    result = generateAnswer(query)
    result = result.replace(query, "").strip()
    start = result.find("[START]") + 7
    end = result.find("[END]")
    script_text = result[start:end].strip()

    keywords = result[end + 5:].strip()
    keywords_start = keywords.find("[KEYWORDS]") + 10
    keywords_end = keywords.find("[KEYWORDSEND]")
    keywords_result = keywords[keywords_start:keywords_end].strip().split(",")

    music_start = result.find("[MUSIC]") + 7
    music_end = result.find("[MUSIC_END]")
    music_keywords = result[music_start:music_end].strip()


    print("Music keywords:", music_keywords)
    print("Keywords: ", keywords_result)

    with open("script.txt", "w", encoding="utf-8") as f:
        f.write(script_text)

    print(script_text)

    # Convert the script text to speech
    tts = gTTS(script_text, slow=False)
    tts.save("script.mp3")

    # Get audio length
    info = sf.info('script.mp3')
    duration_seconds = info.duration

    print("Audio duration (seconds):", duration_seconds)

    # Download videos to match the audio length
    videoDownloader(keywords_result, duration_seconds)

# Setup PyQt application
topic = None

app = QApplication([])
window = QWidget()
Title = QLabel("Video Generator")
text_field = QLineEdit()
button = QPushButton('Run')
button.clicked.connect(Run)

layout = QVBoxLayout()
layout.addWidget(Title)
layout.addWidget(text_field)
layout.addWidget(button)

window.setLayout(layout)
window.show()
app.exec()
