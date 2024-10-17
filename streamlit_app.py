import streamlit as st
import openai
import os
import moviepy.editor as mpe
import requests
from tempfile import NamedTemporaryFile
from datasets import load_dataset
from TTS.api import TTS
from moviepy.video.fx.all import fadein, fadeout, resize
import psutil
from tenacity import retry, wait_random_exponential, stop_after_attempt
import json
from PIL import Image, ImageDraw, ImageFont
import threading
import shutil
from moviepy.video.tools.drawing import color_gradient

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


# 1. Function to generate storyboard based on user prompt using structured JSON
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def generate_storyboard(prompt, style="motivational"):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": (
                "You are a creative assistant that generates detailed video storyboards "
                "based on user prompts. Provide key scenes, narration text for voiceover, "
                "and suggestions for titles and text overlays for each scene in JSON format."
            )},
            {"role": "user", "content": f"Prompt: {prompt}\nStyle: {style}"},
        ],
        response_format={"type": "json_object"}
    )
    try:
        return response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
    except (KeyError, IndexError, TypeError):
        return "{}"

# 2. Function to parse structured JSON storyboard data
def parse_storyboard(storyboard):
    try:
        return json.loads(storyboard).get("scenes", [])
    except json.JSONDecodeError:
        return []

# 3. Function to fetch video clips dynamically based on scene keywords
def fetch_video_clips(scenes):
    dataset = load_dataset('HuggingFaceM4/stock-footage', split='train')
    video_clips = []
    for scene in scenes:
        video_file = search_and_download_video(dataset, scene.get('keywords', 'nature'))
        if video_file:
            video_clips.append({'clip': mpe.VideoFileClip(video_file.name), 'scene': scene})
    return video_clips

# 4. Function to search and download video clips based on keywords
def search_and_download_video(dataset, query):
    for item in dataset:
        if query.lower() in item.get('text', '').lower():
            video_response = requests.get(item['url'])
            if video_response.status_code == 200:
                temp_video_file = NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_video_file.write(video_response.content)
                temp_video_file.flush()
                return temp_video_file
    return None

# 5. Function to generate voiceover with reusable TTS model
def generate_voiceover(narration_text, voice_speed=1.0, pitch=1.0):
    tts_model = initialize_tts_model()
    tts = TTS(tts_model)
    temp_audio_file = NamedTemporaryFile(delete=False, suffix='.wav')
    tts.tts_to_file(text=narration_text, file_path=temp_audio_file.name, speed=voice_speed, pitch=pitch)
    return temp_audio_file.name

# 6. Function to initialize TTS model once
def initialize_tts_model():
    if not hasattr(initialize_tts_model, "tts_model"):
        available_models = TTS.list_models()
        if not available_models:
            raise ValueError("No TTS models available. Please check the library version and internet connection.")
        initialize_tts_model.tts_model = available_models[0]
    return initialize_tts_model.tts_model

# 7. Function to create and finalize the video
def create_video(video_clips, narration_file):
    try:
        final_clip = mpe.concatenate_videoclips([
            apply_fade_effects(add_text_overlay(clip['clip'], clip['scene'].get('overlay', ''))) for clip in video_clips
        ], method='compose')
        final_clip = add_narration(final_clip, narration_file)
        final_clip.write_videofile("final_video.mp4", fps=24, codec='libx264')
    except Exception as e:
        raise RuntimeError(f"Error during video creation: {e}")

# 8. Function to apply fade-in/fade-out effects to video clips
def apply_fade_effects(clip, duration=1):
    try:
        return fadein(clip, duration).fx(fadeout, duration)
    except Exception as e:
        raise ValueError(f"Error applying fade effects: {e}")

# 9. Function to add text overlay to video clips
def add_text_overlay(clip, text):
    if text:
        try:
            text_clip = mpe.TextClip(text, fontsize=70, color='white', font='Arial-Bold')
            text_clip = text_clip.set_position('center').set_duration(clip.duration)
            return mpe.CompositeVideoClip([clip, text_clip])
        except Exception as e:
            raise ValueError(f"Error adding text overlay: {e}")
    return clip

# 10. Function to add narration to video clip
def add_narration(clip, narration_file):
    try:
        return clip.set_audio(mpe.AudioFileClip(narration_file))
    except Exception as e:
        raise ValueError(f"Error adding narration: {e}")

# 11. Function to add background music to video
def add_background_music(clip, music_file):
    try:
        background_audio = mpe.AudioFileClip(music_file)
        return clip.set_audio(mpe.CompositeAudioClip([clip.audio, background_audio.volumex(0.1)]))
    except Exception as e:
        raise ValueError(f"Error adding background music: {e}")

# 12. Function to add watermarks to video clips
def add_watermark(clip, watermark_text="Sample Watermark"):
    try:
        watermark = mpe.TextClip(watermark_text, fontsize=30, color='white', font='Arial')
        watermark = watermark.set_position(('right', 'bottom')).set_duration(clip.duration)
        return mpe.CompositeVideoClip([clip, watermark])
    except Exception as e:
        raise ValueError(f"Error adding watermark: {e}")

# 13. Function to split video into parts for processing
def split_video(video_clip, part_duration=10):
    try:
        return [video_clip.subclip(start, min(start + part_duration, video_clip.duration)) for start in range(0, int(video_clip.duration), part_duration)]
    except Exception as e:
        raise ValueError(f"Error splitting video: {e}")

# 14. Function to merge video parts back together
def merge_video_parts(video_parts):
    try:
        return mpe.concatenate_videoclips(video_parts, method="compose")
    except Exception as e:
        raise ValueError(f"Error merging video parts: {e}")

# 15. Function to save a temporary JSON backup of generated storyboard
def save_storyboard_backup(storyboard, filename="storyboard_backup.json"):
    try:
        with open(filename, 'w') as f:
            json.dump(storyboard, f)
    except Exception as e:
        raise ValueError(f"Error saving storyboard backup: {e}")

# 16. Function to load a saved storyboard from backup
def load_storyboard_backup(filename="storyboard_backup.json"):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading storyboard backup: {e}")

# 17. Function to add subtitles to video
def add_subtitles_to_video(clip, subtitles):
    try:
        subtitle_clips = [
            mpe.TextClip(subtitle['text'], fontsize=50, color='white', size=clip.size, font='Arial-Bold')
            .set_position(('bottom')).set_start(subtitle['start']).set_duration(subtitle['duration'])
            for subtitle in subtitles
        ]
        return mpe.CompositeVideoClip([clip] + subtitle_clips)
    except Exception as e:
        raise ValueError(f"Error adding subtitles: {e}")

# 18. Function to preview storyboard as a slideshow
def preview_storyboard_slideshow(scenes, duration_per_scene=5):
    try:
        slides = [create_animated_text(scene['title'], duration=duration_per_scene) for scene in scenes]
        slideshow = mpe.concatenate_videoclips(slides, method='compose')
        slideshow.write_videofile("storyboard_preview.mp4", fps=24, codec='libx264')
    except Exception as e:
        raise ValueError(f"Error creating storyboard slideshow: {e}")

# 19. Function to add logo to video
def add_logo_to_video(clip, logo_path, position=('right', 'top')):
    try:
        logo = mpe.ImageClip(logo_path).set_duration(clip.duration).resize(height=100).set_position(position)
        return mpe.CompositeVideoClip([clip, logo])
    except Exception as e:
        raise ValueError(f"Error adding logo to video: {e}")

# 20. Function to compress video output for faster uploading
def compress_video(input_path, output_path="compressed_video.mp4", bitrate="500k"):
    try:
        os.system(f"ffmpeg -i {input_path} -b:v {bitrate} -bufsize {bitrate} {output_path}")
    except Exception as e:
        raise ValueError(f"Error compressing video: {e}")

# 21. Function to apply black-and-white filter to video
def apply_bw_filter(clip):
    try:
        return clip.fx(mpe.vfx.blackwhite)
    except Exception as e:
        raise ValueError(f"Error applying black-and-white filter: {e}")

# 22. Function to animate text sequences
def create_animated_text(text, duration=5):
    try:
        txt_clip = mpe.TextClip(text, fontsize=70, color='yellow', font='Arial-Bold', kerning=5)
        txt_clip = txt_clip.set_position('center').set_duration(duration).fadein(1).fadeout(1)
        return txt_clip
    except Exception as e:
        raise ValueError(f"Error creating animated text: {e}")

# 23. Function to overlay images on video
def overlay_image_on_video(clip, image_path, position=(0, 0)):
    try:
        image = mpe.ImageClip(image_path).set_duration(clip.duration).set_position(position)
        return mpe.CompositeVideoClip([clip, image])
    except Exception as e:
        raise ValueError(f"Error overlaying image on video: {e}")

# 24. Function to adjust video speed
def adjust_video_speed(clip, speed=1.0):
    try:
        return clip.fx(mpe.vfx.speedx, speed)
    except Exception as e:
        raise ValueError(f"Error adjusting video speed: {e}")

# 25. Function to crop video clips
def crop_video(clip, x1, y1, x2, y2):
    try:
        return clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)
    except Exception as e:
        raise ValueError(f"Error cropping video: {e}")

# 26. Function to adjust resolution dynamically based on system capacity
def adjust_resolution_based_on_system(clip):
    try:
        memory = psutil.virtual_memory()
        resolution = (640, 360) if memory.available < 1000 * 1024 * 1024 else (1280, 720)
        return resize(clip, newsize=resolution)
    except Exception as e:
        raise ValueError(f"Error adjusting resolution: {e}")

# 27. Function to generate video thumbnail
def generate_video_thumbnail(clip, output_path="thumbnail.png"):
    try:
        frame = clip.get_frame(1)
        image = Image.fromarray(frame)
        image.save(output_path)
        return output_path
    except Exception as e:
        raise ValueError(f"Error generating video thumbnail: {e}")

# 28. Function to animate scene transitions
def animate_scene_transition(clip1, clip2, duration=1):
    try:
        return mpe.concatenate_videoclips([fadeout(clip1, duration), fadein(clip2, duration)])
    except Exception as e:
        raise ValueError(f"Error animating scene transition: {e}")

# 29. Function to add intro and outro sequences to video
def add_intro_outro(clip, intro_text="Welcome", outro_text="Thank you for watching!"):
    try:
        intro = create_animated_text(intro_text, duration=3)
        outro = create_animated_text(outro_text, duration=3)
        return mpe.concatenate_videoclips([intro, clip, outro])
    except Exception as e:
        raise ValueError(f"Error adding intro/outro: {e}")

# 30. Function to adjust audio volume levels
def adjust_audio_volume(audio_clip, volume_level=1.0):
    try:
        return audio_clip.volumex(volume_level)
    except Exception as e:
        raise ValueError(f"Error adjusting audio volume: {e}")

# 31. Function to generate a text overlay with gradient background
def generate_gradient_text_overlay(text, clip_duration, size=(1920, 1080)):
    try:
        gradient = color_gradient(size, p1=(0, 0), p2=(size[0], size[1]), color1=(255, 0, 0), color2=(0, 0, 255))
        image = Image.fromarray(gradient)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        text_size = draw.textsize(text, font=font)
        draw.text(((size[0] - text_size[0]) / 2, (size[1] - text_size[1]) / 2), text, font=font, fill=(255, 255, 255))
        image.save("gradient_overlay.png")
        return mpe.ImageClip("gradient_overlay.png").set_duration(clip_duration)
    except Exception as e:
        raise ValueError(f"Error generating gradient text overlay: {e}")

# 32. Function to run video rendering in a separate thread
def run_video_rendering_thread(target_function, *args):
    try:
        rendering_thread = threading.Thread(target=target_function, args=args)
        rendering_thread.start()
        return rendering_thread
    except Exception as e:
        raise ValueError(f"Error running rendering thread: {e}")

# 33. Function to check system capabilities before rendering
def check_system_capabilities():
    try:
        memory = psutil.virtual_memory()
        if memory.available < 500 * 1024 * 1024:  # Less than 500MB
            st.warning("Low memory detected. Consider closing other applications.")
    except Exception as e:
        st.error(f"Error checking system capabilities: {e}")

# 34. Function to log system resources during video generation
def log_system_resources():
    try:
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        st.write(f"Memory Usage: {memory.percent}% | CPU Usage: {cpu}%")
    except Exception as e:
        st.error(f"Error logging system resources: {e}")

# 35. Function to download additional video assets (e.g., background music)
def download_additional_assets(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            temp_asset_file = NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_asset_file.write(response.content)
            temp_asset_file.flush()
            return temp_asset_file.name
        else:
            raise ValueError("Failed to download asset. Invalid URL or server error.")
    except Exception as e:
        raise ValueError(f"Error downloading asset: {e}")

# 36. Function to calculate estimated video rendering time
def calculate_estimated_render_time(duration, resolution=(1280, 720)):
    try:
        return duration * (resolution[0] * resolution[1]) / 1e6
    except Exception as e:
        raise ValueError(f"Error calculating render time: {e}")

# 37. Function to manage temporary directories
def manage_temp_directory(directory_path):
    try:
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
        os.makedirs(directory_path)
    except Exception as e:
        raise ValueError(f"Error managing temporary directory: {e}")

# 38. Function to handle session expiration or token errors
def handle_session_expiration():
    try:
        st.error("Session expired. Please refresh and try again.")
        st.button("Refresh Page")
    except Exception as e:
        st.error(f"Error handling session expiration: {e}")

# 39. Function to split storyboard scenes for easy preview
def split_storyboard_scenes(scenes, batch_size=5):
    try:
        return [scenes[i:i + batch_size] for i in range(0, len(scenes), batch_size)]
    except Exception as e:
        raise ValueError(f"Error splitting storyboard scenes: {e}")

# 40. Function to add transition effects between storyboard scenes
def add_transition_effects_between_scenes(scenes):
    try:
        return [animate_scene_transition(scene1, scene2) for scene1, scene2 in zip(scenes, scenes[1:])]
    except Exception as e:
        raise ValueError(f"Error adding transition effects: {e}")

# 41. Function to optimize storyboard scene text prompts
def optimize_storyboard_text_prompts(scenes):
    try:
        for scene in scenes:
            scene['title'] = scene['title'].capitalize()
        return scenes
    except Exception as e:
        raise ValueError(f"Error optimizing storyboard text prompts: {e}")

# Main function to run the Streamlit app
def main():
    st.title("🎥 AI-Powered Video Editor")
    prompt = st.text_area("Enter your video idea:", height=150)
    style = st.selectbox("Select a storyboard style", ["motivational", "dramatic", "educational", "funny"])
    voice_speed = st.slider("Select voice speed", 0.5, 2.0, 1.0)
    voice_pitch = st.slider("Select voice pitch", 0.5, 2.0, 1.0)
    add_intro_outro_option = st.checkbox("Add Intro/Outro", value=True)
    add_watermark_option = st.checkbox("Add Watermark", value=True)
    watermark_text = st.text_input("Watermark Text", "Sample Watermark") if add_watermark_option else None
    add_background_music_option = st.checkbox("Add Background Music", value=False)
    music_url = st.text_input("Background Music URL") if add_background_music_option else None

    if st.button("Generate Video"):
        if not prompt.strip():
            st.error("Please enter a valid video idea.")
            return

        try:
            st.info("Generating storyboard...")
            storyboard = generate_storyboard(prompt, style=style)
            if not storyboard:
                st.error("Failed to generate storyboard.")
                return
            st.success("Storyboard generated.")
            st.subheader("Generated Storyboard")
            st.text(storyboard)

            st.info("Parsing storyboard...")
            scenes = parse_storyboard(storyboard)
            if not scenes:
                st.error("Failed to parse storyboard.")
                return
            st.success("Storyboard parsed successfully.")

            narration_text = ' '.join([scene.get('narration', '') for scene in scenes])
            if not narration_text:
                st.error("No narration text found in storyboard.")
                return

            st.info("Fetching video clips...")
            video_clips = fetch_video_clips(scenes)
            if not video_clips:
                st.error("No video clips fetched.")
                return
            st.success("Video clips fetched.")

            st.info("Generating voiceover...")
            narration_file = generate_voiceover(narration_text, voice_speed=voice_speed, pitch=voice_pitch)
            st.success("Voiceover generated.")

            st.info("Creating final video...")
            processed_clips = []
            for clip in video_clips:
                if add_watermark_option:
                    clip['clip'] = add_watermark(clip['clip'], watermark_text)
                processed_clips.append(clip['clip'])

            final_clip = mpe.concatenate_videoclips(processed_clips, method='compose')
            if add_intro_outro_option:
                final_clip = add_intro_outro(final_clip)

            if music_url:
                st.info("Downloading background music...")
                music_file = download_additional_assets(music_url)
                if music_file:
                    final_clip = add_background_music(final_clip, music_file)
                    st.success("Background music added.")
                else:
                    st.warning("Failed to download background music.")

            final_clip.write_videofile("final_video.mp4", fps=24, codec='libx264')
            st.success("Video created successfully!")

            with open("final_video.mp4", 'rb') as video_file:
                st.video(video_file.read())
                st.download_button(label="Download Video", data=video_file, file_name="final_video.mp4")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()import streamlit as st
import openai
import os
import moviepy.editor as mpe
import requests
from tempfile import NamedTemporaryFile
from datasets import load_dataset
from TTS.api import TTS
from moviepy.video.fx.all import fadein, fadeout, resize
import psutil
from tenacity import retry, wait_random_exponential, stop_after_attempt
import json
from PIL import Image, ImageDraw, ImageFont
import threading
import shutil
from moviepy.video.tools.drawing import color_gradient

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


# 1. Function to generate storyboard based on user prompt using structured JSON
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def generate_storyboard(prompt, style="motivational"):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": (
                "You are a creative assistant that generates detailed video storyboards "
                "based on user prompts. Provide key scenes, narration text for voiceover, "
                "and suggestions for titles and text overlays for each scene in JSON format."
            )},
            {"role": "user", "content": f"Prompt: {prompt}\nStyle: {style}"},
        ],
        response_format={"type": "json_object"}
    )
    try:
        return response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
    except (KeyError, IndexError, TypeError):
        return "{}"

# 2. Function to parse structured JSON storyboard data
def parse_storyboard(storyboard):
    try:
        return json.loads(storyboard).get("scenes", [])
    except json.JSONDecodeError:
        return []

# 3. Function to fetch video clips dynamically based on scene keywords
def fetch_video_clips(scenes):
    dataset = load_dataset('HuggingFaceM4/stock-footage', split='train')
    video_clips = []
    for scene in scenes:
        video_file = search_and_download_video(dataset, scene.get('keywords', 'nature'))
        if video_file:
            video_clips.append({'clip': mpe.VideoFileClip(video_file.name), 'scene': scene})
    return video_clips

# 4. Function to search and download video clips based on keywords
def search_and_download_video(dataset, query):
    for item in dataset:
        if query.lower() in item.get('text', '').lower():
            video_response = requests.get(item['url'])
            if video_response.status_code == 200:
                temp_video_file = NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_video_file.write(video_response.content)
                temp_video_file.flush()
                return temp_video_file
    return None

# 5. Function to generate voiceover with reusable TTS model
def generate_voiceover(narration_text, voice_speed=1.0, pitch=1.0):
    tts_model = initialize_tts_model()
    tts = TTS(tts_model)
    temp_audio_file = NamedTemporaryFile(delete=False, suffix='.wav')
    tts.tts_to_file(text=narration_text, file_path=temp_audio_file.name, speed=voice_speed, pitch=pitch)
    return temp_audio_file.name

# 6. Function to initialize TTS model once
def initialize_tts_model():
    if not hasattr(initialize_tts_model, "tts_model"):
        available_models = TTS.list_models()
        if not available_models:
            raise ValueError("No TTS models available. Please check the library version and internet connection.")
        initialize_tts_model.tts_model = available_models[0]
    return initialize_tts_model.tts_model

# 7. Function to create and finalize the video
def create_video(video_clips, narration_file):
    try:
        final_clip = mpe.concatenate_videoclips([
            apply_fade_effects(add_text_overlay(clip['clip'], clip['scene'].get('overlay', ''))) for clip in video_clips
        ], method='compose')
        final_clip = add_narration(final_clip, narration_file)
        final_clip.write_videofile("final_video.mp4", fps=24, codec='libx264')
    except Exception as e:
        raise RuntimeError(f"Error during video creation: {e}")

# 8. Function to apply fade-in/fade-out effects to video clips
def apply_fade_effects(clip, duration=1):
    try:
        return fadein(clip, duration).fx(fadeout, duration)
    except Exception as e:
        raise ValueError(f"Error applying fade effects: {e}")

# 9. Function to add text overlay to video clips
def add_text_overlay(clip, text):
    if text:
        try:
            text_clip = mpe.TextClip(text, fontsize=70, color='white', font='Arial-Bold')
            text_clip = text_clip.set_position('center').set_duration(clip.duration)
            return mpe.CompositeVideoClip([clip, text_clip])
        except Exception as e:
            raise ValueError(f"Error adding text overlay: {e}")
    return clip

# 10. Function to add narration to video clip
def add_narration(clip, narration_file):
    try:
        return clip.set_audio(mpe.AudioFileClip(narration_file))
    except Exception as e:
        raise ValueError(f"Error adding narration: {e}")

# 11. Function to add background music to video
def add_background_music(clip, music_file):
    try:
        background_audio = mpe.AudioFileClip(music_file)
        return clip.set_audio(mpe.CompositeAudioClip([clip.audio, background_audio.volumex(0.1)]))
    except Exception as e:
        raise ValueError(f"Error adding background music: {e}")

# 12. Function to add watermarks to video clips
def add_watermark(clip, watermark_text="Sample Watermark"):
    try:
        watermark = mpe.TextClip(watermark_text, fontsize=30, color='white', font='Arial')
        watermark = watermark.set_position(('right', 'bottom')).set_duration(clip.duration)
        return mpe.CompositeVideoClip([clip, watermark])
    except Exception as e:
        raise ValueError(f"Error adding watermark: {e}")

# 13. Function to split video into parts for processing
def split_video(video_clip, part_duration=10):
    try:
        return [video_clip.subclip(start, min(start + part_duration, video_clip.duration)) for start in range(0, int(video_clip.duration), part_duration)]
    except Exception as e:
        raise ValueError(f"Error splitting video: {e}")

# 14. Function to merge video parts back together
def merge_video_parts(video_parts):
    try:
        return mpe.concatenate_videoclips(video_parts, method="compose")
    except Exception as e:
        raise ValueError(f"Error merging video parts: {e}")

# 15. Function to save a temporary JSON backup of generated storyboard
def save_storyboard_backup(storyboard, filename="storyboard_backup.json"):
    try:
        with open(filename, 'w') as f:
            json.dump(storyboard, f)
    except Exception as e:
        raise ValueError(f"Error saving storyboard backup: {e}")

# 16. Function to load a saved storyboard from backup
def load_storyboard_backup(filename="storyboard_backup.json"):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading storyboard backup: {e}")

# 17. Function to add subtitles to video
def add_subtitles_to_video(clip, subtitles):
    try:
        subtitle_clips = [
            mpe.TextClip(subtitle['text'], fontsize=50, color='white', size=clip.size, font='Arial-Bold')
            .set_position(('bottom')).set_start(subtitle['start']).set_duration(subtitle['duration'])
            for subtitle in subtitles
        ]
        return mpe.CompositeVideoClip([clip] + subtitle_clips)
    except Exception as e:
        raise ValueError(f"Error adding subtitles: {e}")

# 18. Function to preview storyboard as a slideshow
def preview_storyboard_slideshow(scenes, duration_per_scene=5):
    try:
        slides = [create_animated_text(scene['title'], duration=duration_per_scene) for scene in scenes]
        slideshow = mpe.concatenate_videoclips(slides, method='compose')
        slideshow.write_videofile("storyboard_preview.mp4", fps=24, codec='libx264')
    except Exception as e:
        raise ValueError(f"Error creating storyboard slideshow: {e}")

# 19. Function to add logo to video
def add_logo_to_video(clip, logo_path, position=('right', 'top')):
    try:
        logo = mpe.ImageClip(logo_path).set_duration(clip.duration).resize(height=100).set_position(position)
        return mpe.CompositeVideoClip([clip, logo])
    except Exception as e:
        raise ValueError(f"Error adding logo to video: {e}")

# 20. Function to compress video output for faster uploading
def compress_video(input_path, output_path="compressed_video.mp4", bitrate="500k"):
    try:
        os.system(f"ffmpeg -i {input_path} -b:v {bitrate} -bufsize {bitrate} {output_path}")
    except Exception as e:
        raise ValueError(f"Error compressing video: {e}")

# 21. Function to apply black-and-white filter to video
def apply_bw_filter(clip):
    try:
        return clip.fx(mpe.vfx.blackwhite)
    except Exception as e:
        raise ValueError(f"Error applying black-and-white filter: {e}")

# 22. Function to animate text sequences
def create_animated_text(text, duration=5):
    try:
        txt_clip = mpe.TextClip(text, fontsize=70, color='yellow', font='Arial-Bold', kerning=5)
        txt_clip = txt_clip.set_position('center').set_duration(duration).fadein(1).fadeout(1)
        return txt_clip
    except Exception as e:
        raise ValueError(f"Error creating animated text: {e}")

# 23. Function to overlay images on video
def overlay_image_on_video(clip, image_path, position=(0, 0)):
    try:
        image = mpe.ImageClip(image_path).set_duration(clip.duration).set_position(position)
        return mpe.CompositeVideoClip([clip, image])
    except Exception as e:
        raise ValueError(f"Error overlaying image on video: {e}")

# 24. Function to adjust video speed
def adjust_video_speed(clip, speed=1.0):
    try:
        return clip.fx(mpe.vfx.speedx, speed)
    except Exception as e:
        raise ValueError(f"Error adjusting video speed: {e}")

# 25. Function to crop video clips
def crop_video(clip, x1, y1, x2, y2):
    try:
        return clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)
    except Exception as e:
        raise ValueError(f"Error cropping video: {e}")

# 26. Function to adjust resolution dynamically based on system capacity
def adjust_resolution_based_on_system(clip):
    try:
        memory = psutil.virtual_memory()
        resolution = (640, 360) if memory.available < 1000 * 1024 * 1024 else (1280, 720)
        return resize(clip, newsize=resolution)
    except Exception as e:
        raise ValueError(f"Error adjusting resolution: {e}")

# 27. Function to generate video thumbnail
def generate_video_thumbnail(clip, output_path="thumbnail.png"):
    try:
        frame = clip.get_frame(1)
        image = Image.fromarray(frame)
        image.save(output_path)
        return output_path
    except Exception as e:
        raise ValueError(f"Error generating video thumbnail: {e}")

# 28. Function to animate scene transitions
def animate_scene_transition(clip1, clip2, duration=1):
    try:
        return mpe.concatenate_videoclips([fadeout(clip1, duration), fadein(clip2, duration)])
    except Exception as e:
        raise ValueError(f"Error animating scene transition: {e}")

# 29. Function to add intro and outro sequences to video
def add_intro_outro(clip, intro_text="Welcome", outro_text="Thank you for watching!"):
    try:
        intro = create_animated_text(intro_text, duration=3)
        outro = create_animated_text(outro_text, duration=3)
        return mpe.concatenate_videoclips([intro, clip, outro])
    except Exception as e:
        raise ValueError(f"Error adding intro/outro: {e}")

# 30. Function to adjust audio volume levels
def adjust_audio_volume(audio_clip, volume_level=1.0):
    try:
        return audio_clip.volumex(volume_level)
    except Exception as e:
        raise ValueError(f"Error adjusting audio volume: {e}")

# 31. Function to generate a text overlay with gradient background
def generate_gradient_text_overlay(text, clip_duration, size=(1920, 1080)):
    try:
        gradient = color_gradient(size, p1=(0, 0), p2=(size[0], size[1]), color1=(255, 0, 0), color2=(0, 0, 255))
        image = Image.fromarray(gradient)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        text_size = draw.textsize(text, font=font)
        draw.text(((size[0] - text_size[0]) / 2, (size[1] - text_size[1]) / 2), text, font=font, fill=(255, 255, 255))
        image.save("gradient_overlay.png")
        return mpe.ImageClip("gradient_overlay.png").set_duration(clip_duration)
    except Exception as e:
        raise ValueError(f"Error generating gradient text overlay: {e}")

# 32. Function to run video rendering in a separate thread
def run_video_rendering_thread(target_function, *args):
    try:
        rendering_thread = threading.Thread(target=target_function, args=args)
        rendering_thread.start()
        return rendering_thread
    except Exception as e:
        raise ValueError(f"Error running rendering thread: {e}")

# 33. Function to check system capabilities before rendering
def check_system_capabilities():
    try:
        memory = psutil.virtual_memory()
        if memory.available < 500 * 1024 * 1024:  # Less than 500MB
            st.warning("Low memory detected. Consider closing other applications.")
    except Exception as e:
        st.error(f"Error checking system capabilities: {e}")

# 34. Function to log system resources during video generation
def log_system_resources():
    try:
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        st.write(f"Memory Usage: {memory.percent}% | CPU Usage: {cpu}%")
    except Exception as e:
        st.error(f"Error logging system resources: {e}")

# 35. Function to download additional video assets (e.g., background music)
def download_additional_assets(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            temp_asset_file = NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_asset_file.write(response.content)
            temp_asset_file.flush()
            return temp_asset_file.name
        else:
            raise ValueError("Failed to download asset. Invalid URL or server error.")
    except Exception as e:
        raise ValueError(f"Error downloading asset: {e}")

# 36. Function to calculate estimated video rendering time
def calculate_estimated_render_time(duration, resolution=(1280, 720)):
    try:
        return duration * (resolution[0] * resolution[1]) / 1e6
    except Exception as e:
        raise ValueError(f"Error calculating render time: {e}")

# 37. Function to manage temporary directories
def manage_temp_directory(directory_path):
    try:
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
        os.makedirs(directory_path)
    except Exception as e:
        raise ValueError(f"Error managing temporary directory: {e}")

# 38. Function to handle session expiration or token errors
def handle_session_expiration():
    try:
        st.error("Session expired. Please refresh and try again.")
        st.button("Refresh Page")
    except Exception as e:
        st.error(f"Error handling session expiration: {e}")

# 39. Function to split storyboard scenes for easy preview
def split_storyboard_scenes(scenes, batch_size=5):
    try:
        return [scenes[i:i + batch_size] for i in range(0, len(scenes), batch_size)]
    except Exception as e:
        raise ValueError(f"Error splitting storyboard scenes: {e}")

# 40. Function to add transition effects between storyboard scenes
def add_transition_effects_between_scenes(scenes):
    try:
        return [animate_scene_transition(scene1, scene2) for scene1, scene2 in zip(scenes, scenes[1:])]
    except Exception as e:
        raise ValueError(f"Error adding transition effects: {e}")

# 41. Function to optimize storyboard scene text prompts
def optimize_storyboard_text_prompts(scenes):
    try:
        for scene in scenes:
            scene['title'] = scene['title'].capitalize()
        return scenes
    except Exception as e:
        raise ValueError(f"Error optimizing storyboard text prompts: {e}")

# Main function to run the Streamlit app
def main():
    st.title("🎥 AI-Powered Video Editor")
    prompt = st.text_area("Enter your video idea:", height=150)
    style = st.selectbox("Select a storyboard style", ["motivational", "dramatic", "educational", "funny"])
    voice_speed = st.slider("Select voice speed", 0.5, 2.0, 1.0)
    voice_pitch = st.slider("Select voice pitch", 0.5, 2.0, 1.0)
    add_intro_outro_option = st.checkbox("Add Intro/Outro", value=True)
    add_watermark_option = st.checkbox("Add Watermark", value=True)
    watermark_text = st.text_input("Watermark Text", "Sample Watermark") if add_watermark_option else None
    add_background_music_option = st.checkbox("Add Background Music", value=False)
    music_url = st.text_input("Background Music URL") if add_background_music_option else None

    if st.button("Generate Video"):
        if not prompt.strip():
            st.error("Please enter a valid video idea.")
            return

        try:
            st.info("Generating storyboard...")
            storyboard = generate_storyboard(prompt, style=style)
            if not storyboard:
                st.error("Failed to generate storyboard.")
                return
            st.success("Storyboard generated.")
            st.subheader("Generated Storyboard")
            st.text(storyboard)

            st.info("Parsing storyboard...")
            scenes = parse_storyboard(storyboard)
            if not scenes:
                st.error("Failed to parse storyboard.")
                return
            st.success("Storyboard parsed successfully.")

            narration_text = ' '.join([scene.get('narration', '') for scene in scenes])
            if not narration_text:
                st.error("No narration text found in storyboard.")
                return

            st.info("Fetching video clips...")
            video_clips = fetch_video_clips(scenes)
            if not video_clips:
                st.error("No video clips fetched.")
                return
            st.success("Video clips fetched.")

            st.info("Generating voiceover...")
            narration_file = generate_voiceover(narration_text, voice_speed=voice_speed, pitch=voice_pitch)
            st.success("Voiceover generated.")

            st.info("Creating final video...")
            processed_clips = []
            for clip in video_clips:
                if add_watermark_option:
                    clip['clip'] = add_watermark(clip['clip'], watermark_text)
                processed_clips.append(clip['clip'])

            final_clip = mpe.concatenate_videoclips(processed_clips, method='compose')
            if add_intro_outro_option:
                final_clip = add_intro_outro(final_clip)

            if music_url:
                st.info("Downloading background music...")
                music_file = download_additional_assets(music_url)
                if music_file:
                    final_clip = add_background_music(final_clip, music_file)
                    st.success("Background music added.")
                else:
                    st.warning("Failed to download background music.")

            final_clip.write_videofile("final_video.mp4", fps=24, codec='libx264')
            st.success("Video created successfully!")

            with open("final_video.mp4", 'rb') as video_file:
                st.video(video_file.read())
                st.download_button(label="Download Video", data=video_file, file_name="final_video.mp4")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
