import scipy.signal
import tensorflow as tf
import tensorflow_hub as hub
import csv
import urllib.request
import datetime
import io
import librosa
import numpy as np
from googletrans import Translator
from deep_translator import GoogleTranslator
from datetime import datetime
import pytz
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Initialize Firebase Admin (do this only once)
cred = credentials.Certificate('./services/soundaware.json')  # <- Change this path
firebase_admin.initialize_app(cred)
db = firestore.client()

class SoundClassData:
    # Lista de alarmas / sonidos de emergencia
    ALARM_CLASSES = [
        "Fire alarm", "Smoke detector", "Siren", "Civil defense siren", 
        "Ambulance (siren)", "Fire engine, fire truck (siren)", "Police car (siren)", 
        "Foghorn", "Alarm", "Gunshot, gunfire", "Machine gun", "Fusillade", 
        "Artillery fire", "Explosion", "Boom", "Screaming", "Alarm clock",
        "Car alarm", "Buzzer", "Emergency vehicle", "Telephone bell ringing",
        "Ringtone", "Baby cry, infant cry", "Shatter", "Breaking", "Doorbell",
        "Ding-dong", "Knock", "Beep, bleep", "Toot", "Air horn, truck horn",
        "Reversing beeps", "Boiling", "Gush", "Burst, pop", "Blender",
        "Whimper", "Wail, moan", "Vehicle horn, car horn, honking",
        "Thunderstorm", "Thunder", "Microwave oven", "Water tap, faucet",
        "Sink (filling or washing)", "Engine starting", "Bark", "Meow",
        "Bicycle bell", "Bell"
    ]

    # Mapeos de urgencia
    URGENCY_MAP = {
        "Fire alarm": "ALTA",
        "Smoke detector": "ALTA",
        "Siren": "ALTA",
        "Civil defense siren": "ALTA",
        "Ambulance (siren)": "ALTA",
        "Fire engine, fire truck (siren)": "ALTA",
        "Police car (siren)": "ALTA",
        "Foghorn": "ALTA",
        "Alarm": "ALTA",
        "Gunshot, gunfire": "ALTA",
        "Machine gun": "ALTA",
        "Fusillade": "ALTA",
        "Artillery fire": "ALTA",
        "Explosion": "ALTA",
        "Boom": "ALTA",
        "Screaming": "ALTA",
        "Shatter": "ALTA",
        "Breaking": "ALTA",
        "Emergency vehicle": "ALTA",
        "Doorbell": "MEDIA",
        "Ding-dong": "MEDIA",
        "Knock": "MEDIA",
        "Alarm clock": "MEDIA",
        "Telephone bell ringing": "MEDIA",
        "Ringtone": "MEDIA",
        "Beep, bleep": "MEDIA",
        "Baby cry, infant cry": "MEDIA",
        "Vehicle horn, car horn, honking": "MEDIA",
        "Toot": "MEDIA",
        "Air horn, truck horn": "MEDIA",
        "Reversing beeps": "MEDIA",
        "Car alarm": "MEDIA",
        "Thunderstorm": "MEDIA",
        "Thunder": "MEDIA",
        "Boiling": "MEDIA",
        "Gush": "MEDIA",
        "Burst, pop": "MEDIA",
        "Buzzer": "MEDIA",
        "Blender": "MEDIA",
        "Whimper": "MEDIA",
        "Wail, moan": "MEDIA",
        "Microwave oven": "BAJA",
        "Water tap, faucet": "BAJA",
        "Sink (filling or washing)": "BAJA",
        "Bark": "BAJA",
        "Meow": "BAJA",
        "Engine starting": "BAJA",
        "Bicycle bell": "BAJA",
        "Bell": "BAJA",
    }

    SOUND_DESCRIPTIONS = {
        "Fire alarm": "Alarma de incendio detectada",
        "Smoke detector": "Detector de humo activado",
        "Civil defense siren": "Sirena de emergencia civil",
        "Ambulance (siren)": "Sirena de ambulancia cercana",
        "Fire engine, fire truck (siren)": "Sirena de camión de bomberos",
        "Police car (siren)": "Sirena de vehículo policial",
        "Siren": "Sirena de emergencia",
        "Alarm": "Alarma general activada",
        "Shatter": "Cristal roto detectado",
        "Breaking": "Objeto rompiéndose",
        "Doorbell": "Timbre de puerta",
        "Ding-dong": "Timbre de casa",
        "Knock": "Alguien golpea la puerta",
        "Alarm clock": "Alarma de reloj",
        "Foghorn": "Sirena de niebla",
        "Machine gun": "Ametralladora",
        "Emergency vehicle": "Vehículo de emergencia",
        "Fusillade": "Fusilamiento",
        "Artillery fire": "Fuego de artillería",
        "Beep, bleep": "Bip bip",
        "Toot": "Bocina de vehículo",
        "Air horn, truck horn": "Bocina de aire, bocina de camión",
        "Reversing beeps": "Pitidos de marcha atrás",
        "Boiling": "Ebullición",
        "Gush": "Chorro de agua",
        "Burst, pop": "Ráfaga, estallido",
        "Buzzer": "Zumbido",
        "Blender": "Mezclador",
        "Whimper": "Gimoteo",
        "Wail, moan": "Lamento de una persona",
        "Telephone bell ringing": "Teléfono sonando",
        "Ringtone": "Teléfono móvil sonando",
        "Baby cry, infant cry": "Llanto de bebé detectado",
        "Vehicle horn, car horn, honking": "Bocina de vehículo",
        "Car alarm": "Alarma de auto activada",
        "Gunshot, gunfire": "Disparo detectado",
        "Explosion": "Explosión detectada",
        "Boom": "Explosión o estruendo fuerte",
        "Thunderstorm": "Tormenta eléctrica",
        "Thunder": "Trueno detectado",
        "Screaming": "Grito detectado",
        "Microwave oven": "Microondas sonando",
        "Water tap, faucet": "Agua corriendo",
        "Sink (filling or washing)": "Lavabo llenándose",
        "Engine starting": "Motor encendiéndose",
        "Bark": "Ladrido de perro",
        "Meow": "Maullido de gato",
        "Bicycle bell": "Timbre de bicicleta",
        "Bell": "Campana sonando",
    }

class AudioProcessor:
    translator = Translator()

    def __init__(self, desired_sample_rate=16000):
        self.desired_sample_rate = desired_sample_rate
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        self.class_names = self._load_class_names()
        self.sound_data = SoundClassData()

    def _load_class_names(self):
        class_map_path = self.model.class_map_path().numpy().decode('utf-8')
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
        waveform, sample_rate = tf.audio.decode_wav(audio_bytes)
        sample_rate = sample_rate.numpy()
        
        if waveform.shape[1] > 1:
            waveform = tf.reduce_mean(waveform, axis=1, keepdims=True)
        
        _, waveform = self.ensure_sample_rate(sample_rate, waveform)
        return tf.squeeze(waveform, axis=-1)

    def _process_mp3(self, audio_bytes):
        audio_bytes_io = io.BytesIO(audio_bytes)
        waveform, sample_rate = librosa.load(audio_bytes_io, sr=self.desired_sample_rate, mono=True)
        return tf.convert_to_tensor(waveform, dtype=tf.float32)

    def is_alarm_sound(self, predicted_class, context_sounds, confidence_score):
        if predicted_class in self.sound_data.ALARM_CLASSES and confidence_score >= 0.3:
            return True
        
        alarm_count = sum(1 for sound in context_sounds if sound in self.sound_data.ALARM_CLASSES)
        if alarm_count >= 2:
            return True
            
        return False

    def find_highest_confidence_alarm(self, scores_np_mean, class_names):
        max_score = 0
        max_class = None
        
        for i, class_name in enumerate(class_names):
            if class_name in self.sound_data.ALARM_CLASSES and scores_np_mean[i] > max_score:
                max_score = scores_np_mean[i]
                max_class = class_name
        
        return max_class, max_score if max_class else None

    def analyze_repetition_pattern(self, scores, top_class_idx, time_window=5):
        segment_scores = scores[:min(time_window, scores.shape[0])]
        top_classes = np.argmax(segment_scores, axis=1)
        matches = np.sum(top_classes == top_class_idx)
        match_ratio = matches / len(top_classes)
        
        if match_ratio > 0.8:
            return "repeating_fast"
        elif match_ratio > 0.5:
            return "repeating_slow"
        else:
            return "single"

    def get_urgency_level(self, sound_class, volume_level, repetition_pattern, context_sounds):
        base_urgency = self.sound_data.URGENCY_MAP.get(sound_class, "MONITOREO")
        
        if volume_level > 0.8 and base_urgency in ["MEDIA", "BAJA", "MONITOREO"]:
            urgency_map = {"MEDIA": "ALTA", "BAJA": "MEDIA", "MONITOREO": "BAJA"}
            adjusted_urgency = urgency_map.get(base_urgency, base_urgency)
        elif volume_level < 0.3 and base_urgency in ["ALTA", "MEDIA"]:
            urgency_map = {"ALTA": "MEDIA", "MEDIA": "BAJA"}
            adjusted_urgency = urgency_map.get(base_urgency, base_urgency)
        else:
            adjusted_urgency = base_urgency
        
        if repetition_pattern == "repeating_fast" and adjusted_urgency in ["MEDIA", "BAJA"]:
            urgency_map = {"MEDIA": "ALTA", "BAJA": "MEDIA"}
            adjusted_urgency = urgency_map.get(adjusted_urgency, adjusted_urgency)
        
        has_emergency_context = any(sound in self.sound_data.ALARM_CLASSES for sound in context_sounds)
        if has_emergency_context and adjusted_urgency in ["MEDIA", "BAJA"]:
            urgency_map = {"MEDIA": "ALTA", "BAJA": "MEDIA"}
            adjusted_urgency = urgency_map.get(adjusted_urgency, adjusted_urgency)
        
        confidence = 0.6
        
        if volume_level > 0.7:
            confidence += 0.2
        elif volume_level < 0.3:
            confidence -= 0.1
        
        if has_emergency_context and sound_class in self.sound_data.ALARM_CLASSES:
            confidence += 0.2
        
        confidence = max(0.1, min(0.99, confidence))
        description = self.sound_data.SOUND_DESCRIPTIONS.get(sound_class, sound_class)
        
        return adjusted_urgency, confidence, description

    def _convert_to_native_types(self, data):
        """Convierte los valores numéricos a tipos nativos de Python para serialización JSON"""
        if isinstance(data, (np.generic, np.ndarray)):
            return data.item() if data.size == 1 else data.tolist()
        elif isinstance(data, (tf.Tensor, tf.Variable)):
            return data.numpy().item()
        elif isinstance(data, dict):
            return {k: self._convert_to_native_types(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._convert_to_native_types(item) for item in data]
        return data

    async def process_audio_file(self, audio_file, user_id=None, location=None):
        try:
            audio_bytes = audio_file.read()
            is_mp3 = False
            
            if len(audio_bytes) > 2:
                if audio_bytes[:3] == b'ID3' or (audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0):
                    is_mp3 = True
            
            if is_mp3:
                waveform = self._process_mp3(audio_bytes)
            else:
                try:
                    waveform = self._process_wav(audio_bytes)
                except tf.errors.InvalidArgumentError:
                    waveform = self._process_mp3(audio_bytes)
            
            # Calcular volumen
            waveform_np = waveform.numpy()
            volume_level = float(min(1.0, np.mean(np.abs(waveform_np)) / 0.05))
            
            # Realizar predicción
            scores, embeddings, spectrogram = self.model(waveform)
            scores_np = scores.numpy()
            scores_np_mean = scores_np.mean(axis=0)
            
            # Obtener las 5 clases principales
            top_indices = np.argsort(scores_np_mean)[-5:][::-1]
            top_index = top_indices[0]
            predicted_class = self.class_names[top_index]
            confidence_score = float(scores_np_mean[top_index])
            translated_class = GoogleTranslator(source='en', target='es').translate(predicted_class)
            
            # Sonidos de contexto
            context_sounds = [self.class_names[idx] for idx in top_indices]
            
            # Verificar si el sonido es una alarma
            is_alarm = self.is_alarm_sound(predicted_class, context_sounds, confidence_score)
            
            result = {
                "is_alarm": is_alarm,
                "class": translated_class,
                "original_class": predicted_class,
                "confidence": confidence_score,
                "context_sounds": context_sounds,
                "date": self.date_alert(),
                "volume_level": volume_level,
                "timestamp": datetime.now(pytz.timezone("America/Mexico_City")).isoformat()
            }

            if location:
                result["location"] = location

            if not is_alarm:
                result["message"] = "No se detectó sonido de alarma"
            else:
                if predicted_class not in self.sound_data.ALARM_CLASSES:
                    best_alarm_class, best_alarm_confidence = self.find_highest_confidence_alarm(scores_np_mean, self.class_names)
                    if best_alarm_class:
                        predicted_class = best_alarm_class
                        confidence_score = float(best_alarm_confidence)
                
                repetition_pattern = self.analyze_repetition_pattern(scores_np, top_index)
                
                urgency_level, urgency_confidence, description = self.get_urgency_level(
                    predicted_class, volume_level, repetition_pattern, context_sounds
                )
                
                translated_date = self.translator.translate(self.date_alert(), src='en', dest='es').text
                
                result.update({
                    "urgency_level": urgency_level,
                    "urgency_confidence": float(urgency_confidence),
                    "description": description,
                    "repetition_pattern": repetition_pattern,
                    "translated_date": translated_date,
                    "message": f"Alarma detectada: {description}"
                })

                # Save to Firebase if it's an alarm
                if user_id:
                    alarm_data = {
                        "userId": user_id,
                        "soundClass": predicted_class,
                        "translatedClass": translated_class,
                        "confidence": confidence_score,
                        "urgencyLevel": urgency_level,
                        "description": description,
                        "timestamp": firestore.SERVER_TIMESTAMP,
                        "location": location if location else None,
                        "volumeLevel": volume_level,
                        "status": "pending"
                    }
                    db.collection("alarms").add(alarm_data)
            
            return self._convert_to_native_types(result)
            
        except Exception as e:
            error_msg = f"Error procesando archivo de audio: {str(e)}"
            if user_id:
                error_data = {
                    "userId": user_id,
                    "error": error_msg,
                    "timestamp": firestore.SERVER_TIMESTAMP
                }
                db.collection("error_logs").add(error_data)
            raise ValueError(error_msg)

    def date_alert(self):  
        tz = pytz.timezone("America/Mexico_City")
        x = datetime.now(tz)
        date = x.strftime("%d") + " " + x.strftime("%B") + " of " + x.strftime("%Y") + ", " + x.strftime("%I") + ":" + x.strftime("%M") + x.strftime("%p")
        return date