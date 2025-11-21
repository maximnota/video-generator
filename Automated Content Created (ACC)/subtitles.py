import whisper
import os
import cv2
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip
from moviepy.config import change_settings
from textwrap import wrap

class VideoTranscriber:
    def __init__(self, model_name, video_path, font_path):
        self.model = whisper.load_model(model_name)
        self.video_path = video_path
        self.audio_path = 'script.mp3'
        self.text_array = []
        self.font_path = font_path

    def extract_audio(self):
        print('Extracting audio from video')
        try:
            video = VideoFileClip(self.video_path)
            audio_clip = video.audio
            audio_clip.write_audiofile(self.audio_path)
            print('Audio extracted')
        except Exception as e:
            print(f"Error extracting audio: {e}")

    def transcribe_video(self):
        print('Transcribing video')
        try:
            self.extract_audio()
            result = self.model.transcribe(self.audio_path)

            for segment in result["segments"]:
                text = segment["text"]
                start = segment["start"]
                end = segment["end"]
                self.text_array.append({"text": text, "start": start, "end": end})

            print('Transcription complete')
        except Exception as e:
            print(f"Error during transcription: {e}")

    def create_video(self, output_video_path):
        print('Creating video with subtitles')
        try:
            video = VideoFileClip(self.video_path)
            txt_clips = []
            video_width, video_height = video.size

            for subtitle in self.text_array:
                text = subtitle["text"]
                start = subtitle["start"]
                end = subtitle["end"]

                # Wrap text to fit the width of the video
                font_size = 65
                char_width = font_size * 0.6  # Average width of a character, may need adjustment
                max_chars_per_line = int((video_width - 40) / char_width)  # 20px margin each side
                wrapped_lines = wrap(text, width=max_chars_per_line)
                wrapped_text = "\n".join(wrapped_lines)

                txt_clip = TextClip(wrapped_text, fontsize=font_size, font=self.font_path, color='white', 
                                    stroke_color='black', stroke_width=2)
                txt_clip = txt_clip.set_position(('center', 'center')).set_start(start).set_duration(end - start)
                txt_clips.append(txt_clip)

            video = CompositeVideoClip([video, *txt_clips])
            audio = AudioFileClip(self.audio_path)
            final_clip = video.set_audio(audio)
            final_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

            print('Video created')
        except Exception as e:
            print(f"Error creating video: {e}")

