import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import datetime
import argparse

class AudioToVTT:
    def __init__(self, audio_path, video_path=None):
        self.audio_path = audio_path
        self.video_path = video_path
        self.recognizer = sr.Recognizer()
        
    def check_files(self):
        """Check if input files exist"""
        if self.video_path and not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        if not self.video_path and not os.path.exists(self.audio_path):
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}")
            
    def convert_time(self, milliseconds):
        """Convert milliseconds to WebVTT timestamp format"""
        seconds = milliseconds / 1000
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
        
    def extract_audio(self):
        """Extract audio from video if video path is provided"""
        if self.video_path and self.video_path.endswith(('.mp4', '.avi', '.mov')):
            print(f"Extracting audio from {self.video_path}")
            video = AudioSegment.from_file(self.video_path)
            audio = video.set_channels(1).set_frame_rate(16000)
            audio.export(self.audio_path, format="wav")
            print(f"Audio extracted to {self.audio_path}")
            
    def process_audio(self):
        """Process audio file and generate VTT subtitles"""
        print(f"Processing audio file: {self.audio_path}")
        # Load the audio file
        audio = AudioSegment.from_wav(self.audio_path)
        
        # Split audio on silence
        print("Splitting audio on silence...")
        chunks = split_on_silence(
            audio,
            min_silence_len=500,  # minimum silence length in ms
            silence_thresh=-40,    # silence threshold in dB
            keep_silence=300       # keep some silence for natural breaks
        )
        
        print(f"Found {len(chunks)} audio segments")
        vtt_content = ["WEBVTT\n"]
        current_time = 0
        
        # Process each chunk
        for i, chunk in enumerate(chunks, 1):
            print(f"\rProcessing segment {i}/{len(chunks)}", end="", flush=True)
            # Export chunk to temporary file
            chunk_path = f"temp_chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")
            
            # Recognize speech in chunk
            with sr.AudioFile(chunk_path) as source:
                audio_data = self.recognizer.record(source)
                try:
                    text = self.recognizer.recognize_google(audio_data)
                    if text:
                        # Calculate timestamps
                        start_time = self.convert_time(current_time)
                        end_time = self.convert_time(current_time + len(chunk))
                        
                        # Add subtitle entry
                        vtt_content.extend([
                            f"\n{i}",
                            f"{start_time} --> {end_time}",
                            f"{text}\n"
                        ])
                        
                        current_time += len(chunk)
                except sr.UnknownValueError:
                    print(f"\nNo speech detected in segment {i}")
                except sr.RequestError as e:
                    print(f"\nError with speech recognition service; {e}")
            
            # Clean up temporary file
            os.remove(chunk_path)
        
        print("\nAudio processing completed")
        return "\n".join(vtt_content)
    
    def generate_vtt(self, output_path):
        """Generate VTT file from audio/video"""
        try:
            # Check if input files exist
            self.check_files()
            
            if self.video_path:
                self.extract_audio()
            
            vtt_content = self.process_audio()
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(vtt_content)
            
            print(f"\nVTT file generated successfully at {output_path}")
            
        except Exception as e:
            print(f"Error generating VTT file: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Convert audio/video to VTT subtitles')
    parser.add_argument('input', help='Input audio or video file')
    parser.add_argument('output', help='Output VTT file')
    parser.add_argument('--type', choices=['audio', 'video'], help='Specify input type (audio/video)')
    
    args = parser.parse_args()
    
    try:
        if args.type == 'video':
            converter = AudioToVTT("extracted_audio.wav", args.input)
        else:
            converter = AudioToVTT(args.input)
            
        converter.generate_vtt(args.output)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()