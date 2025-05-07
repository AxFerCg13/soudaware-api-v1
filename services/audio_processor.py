import scipy.signal
import tensorflow as tf
import tensorflow_hub as hub
import csv
import urllib.request
import datetime

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
    def process_audio_file(self, audio_file):
        # Decode wav file
        audio_bytes = audio_file.read()
        waveform, sample_rate = tf.audio.decode_wav(audio_bytes)
        sample_rate = sample_rate.numpy()
        
        # Convert to mono
        if waveform.shape[1] > 1:
            waveform = tf.reduce_mean(waveform, axis=1, keepdims=True)
        
        _, waveform = self.ensure_sample_rate(sample_rate, waveform)
        
        waveform = tf.squeeze(waveform, axis=-1)
        
        # Do prediction
        scores, embeddings, spectrogram = self.model(waveform)
        
        mean_scores = tf.reduce_mean(scores, axis=0)
        top_class = tf.argmax(mean_scores)
        
        return {
            "class": self.class_names[top_class],
            "confidence": mean_scores[top_class].numpy().item(),
            "date": self.date_alert() 
        }

    def date_alert(self):  
        x = datetime.datetime.now()
        date = x.strftime("%w") + " " + x.strftime("%B") + " of " + x.strftime("%Y") + ", "+  x.strftime("%I") + ":" + x.strftime("%M") + x.strftime("%p")
        return date