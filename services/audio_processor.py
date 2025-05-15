import scipy.signal
import tensorflow as tf
import tensorflow_hub as hub
import csv
import urllib.request
import datetime
import io
import librosa
import numpy as np

class AudioProcessor:
    def __init__(self, desired_sample_rate=16000):
        self.desired_sample_rate = desired_sample_rate
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        self.class_names = self._load_class_names()

    def _load_class_names(self):
        # Get class_map model url 
        class_map_path = self.model.class_map_path().numpy().decode('utf-8')
        
        # Download class_map file if it doesn exist
        try:
            with tf.io.gfile.GFile(class_map_path) as csvfile:
                return [row['display_name'] for row in csv.DictReader(csvfile)]
        except:
            class_map_url = "https://storage.googleapis.com/tfhub-models/google/yamnet/1/class_map.csv"
            with urllib.request.urlopen(class_map_url) as response:
                csv_content = response.read().decode('utf-8')
                return [row['display_name'] for row in csv.DictReader(csv_content.splitlines())]

    def ensure_sample_rate(self, original_sample_rate, waveform):
        if original_sample_rate != self.desired_sample_rate:
            desired_length = int(round(float(len(waveform)) / original_sample_rate * self.desired_sample_rate))
            waveform = scipy.signal.resample(waveform, desired_length)
        return self.desired_sample_rate, waveform

    def _process_wav(self, audio_bytes):
        """Procesa archivos WAV (funcionalidad original)"""
        waveform, sample_rate = tf.audio.decode_wav(audio_bytes)
        sample_rate = sample_rate.numpy()
        
        # Convert to mono
        if waveform.shape[1] > 1:
            waveform = tf.reduce_mean(waveform, axis=1, keepdims=True)
        
        _, waveform = self.ensure_sample_rate(sample_rate, waveform)
        return tf.squeeze(waveform, axis=-1)

    def _process_mp3(self, audio_bytes):
        """Procesa archivos MP3"""
        audio_bytes_io = io.BytesIO(audio_bytes)
        waveform, sample_rate = librosa.load(audio_bytes_io, sr=self.desired_sample_rate, mono=True)
        
        # Convertir a tensor de TensorFlow
        waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
        return waveform

    def process_audio_file(self, audio_file):
        try:
            # Leer los bytes del archivo directamente como binario
            audio_bytes = audio_file.read()
            
            # Verificar si es MP3 por los primeros bytes (magic numbers)
            is_mp3 = False
            if len(audio_bytes) > 2:
                # MP3 puede comenzar con ID3 (para metadata) o 0xFF 0xFB (frame sync)
                if audio_bytes[:3] == b'ID3' or (audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0):
                    is_mp3 = True
            
            # Procesar según el tipo de archivo
            if is_mp3:
                waveform = self._process_mp3(audio_bytes)
            else:
                try:
                    waveform = self._process_wav(audio_bytes)
                except tf.errors.InvalidArgumentError:
                    waveform = self._process_mp3(audio_bytes)
            
            scores, embeddings, spectrogram = self.model(waveform)
            mean_scores = tf.reduce_mean(scores, axis=0)
            top_class = tf.argmax(mean_scores)
            
            return {
                "class": self.class_names[top_class],
                "confidence": mean_scores[top_class].numpy().item(),
                "date": self.date_alert() 
            }
        except Exception as e:
            raise ValueError(f"Error procesando archivo de audio: {str(e)}")

    def date_alert(self):  
        x = datetime.datetime.now()
        date = x.strftime("%w") + " " + x.strftime("%B") + " of " + x.strftime("%Y") + ", "+  x.strftime("%I") + ":" + x.strftime("%M") + x.strftime("%p")
        return date