import os
import tempfile
from datetime import datetime
import numpy as np
import csv
import logging
import tensorflow as tf
from google.cloud import storage, firestore
import librosa
import functools

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales para mantener el modelo cargado entre invocaciones
_model = None
_class_names = None

class SoundClassData:
    # Lista de alarmas / sonidos de emergencia (sin cambios)
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

    # Mapeos y descripciones como constantes de clase para mejor rendimiento
    URGENCY_MAP = {
        # URGENCIA ALTA (Requiere Atención Inmediata)
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

        # URGENCIA MEDIA (Requiere Atención Pronta)
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

        # URGENCIA BAJA (Requiere Atención Eventual)
        "Microwave oven": "BAJA",
        "Water tap, faucet": "BAJA",
        "Sink (filling or washing)": "BAJA",
        "Bark": "BAJA",
        "Meow": "BAJA",
        "Engine starting": "BAJA",
        "Bicycle bell": "BAJA",
        "Bell": "BAJA",
        # Por defecto, otros sonidos son de MONITOREO
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

    # Lista de sonidos considerados de emergencia para análisis de contexto
    EMERGENCY_SOUNDS = frozenset([
        "Fire alarm", "Smoke detector", "Siren", "Civil defense siren", 
        "Ambulance (siren)", "Fire engine, fire truck (siren)", "Police car (siren)", 
        "Alarm", "Gunshot, gunfire", "Explosion", "Boom", "Screaming", "Shatter"
    ])


class AudioUtility:
    @staticmethod
    def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
        if original_sample_rate != desired_sample_rate:
            import scipy.signal
            desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
            waveform = scipy.signal.resample(waveform, desired_length)
        return desired_sample_rate, waveform


def load_model():
    global _model, _class_names
    
    if _model is None:
        import tensorflow_hub as hub
        # Cargar modelo
        yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
        _model = hub.load(yamnet_model_handle)
        
        # Cargar nombres de clases
        _class_names = []
        with tf.io.gfile.GFile(_model.class_map_path().numpy()) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                _class_names.append(row['display_name'])
    
    return _model, _class_names


def predict(waveform_tensor):
    model, _ = load_model()
    scores, embeddings, spectrogram = model(waveform_tensor)
    return scores, embeddings, spectrogram


class SoundAnalyzer:
    def __init__(self):
        self.sound_data = SoundClassData()
    
    def is_alarm_sound(self, predicted_class, context_sounds, confidence_score):
        # Verificar si la clase principal es una alarma
        if predicted_class in self.sound_data.ALARM_CLASSES and confidence_score >= 0.3:
            return True
        
        # Verificar si hay 2 sonidos de alarma en el contexto
        alarm_count = sum(1 for sound in context_sounds if sound in self.sound_data.ALARM_CLASSES)
        if alarm_count >= 2:
            return True
            
        return False
    
    def find_highest_confidence_alarm(self, scores_np_mean, class_names):
        # Optimización: buscar directamente el índice de máxima confianza para clases de alarma
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
        
        # Clasificar el patrón de repetición
        if match_ratio > 0.8:
            return "repeating_fast"
        elif match_ratio > 0.5:
            return "repeating_slow"
        else:
            return "single"
    
    def get_urgency_level(self, sound_class, volume_level, repetition_pattern, context_sounds):
        # Nivel base de urgencia
        base_urgency = self.sound_data.URGENCY_MAP.get(sound_class, "MONITOREO")
        
        # Lógica de ajuste de urgencia (optimizada para velocidad)
        if volume_level > 0.8 and base_urgency in ["MEDIA", "BAJA", "MONITOREO"]:
            # Incrementar urgencia si el volumen es muy alto
            urgency_map = {"MEDIA": "ALTA", "BAJA": "MEDIA", "MONITOREO": "BAJA"}
            adjusted_urgency = urgency_map.get(base_urgency, base_urgency)
        elif volume_level < 0.3 and base_urgency in ["ALTA", "MEDIA"]:
            # Disminuir urgencia si el volumen es muy bajo
            urgency_map = {"ALTA": "MEDIA", "MEDIA": "BAJA"}
            adjusted_urgency = urgency_map.get(base_urgency, base_urgency)
        else:
            adjusted_urgency = base_urgency
        
        # Ajustar urgencia basado en patrón de repetición
        if repetition_pattern == "repeating_fast" and adjusted_urgency in ["MEDIA", "BAJA"]:
            urgency_map = {"MEDIA": "ALTA", "BAJA": "MEDIA"}
            adjusted_urgency = urgency_map.get(adjusted_urgency, adjusted_urgency)
        
        # Ajustar basado en contexto (sonidos recientes)
        has_emergency_context = any(sound in self.sound_data.EMERGENCY_SOUNDS for sound in context_sounds)
        if has_emergency_context and adjusted_urgency in ["MEDIA", "BAJA"]:
            urgency_map = {"MEDIA": "ALTA", "BAJA": "MEDIA"}
            adjusted_urgency = urgency_map.get(adjusted_urgency, adjusted_urgency)
        
        # Confianza basada en factores combinados
        confidence = 0.6
        
        # Ajuste confianza basada en volumen
        if volume_level > 0.7:
            confidence += 0.2
        elif volume_level < 0.3:
            confidence -= 0.1
        
        if has_emergency_context and sound_class in self.sound_data.EMERGENCY_SOUNDS:
            confidence += 0.2
        
        confidence = max(0.1, min(0.99, confidence))
        
        description = self.sound_data.SOUND_DESCRIPTIONS.get(sound_class, sound_class)
        
        return adjusted_urgency, confidence, description


# Singleton para clientes
_storage_client = None
_firestore_client = None

def get_storage_client():
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client

def get_firestore_client():
    global _firestore_client
    if _firestore_client is None:
        _firestore_client = firestore.Client()
    return _firestore_client


def download_file(bucket_name, file_name):
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    
    # Usar una extensión basada en el nombre del archivo
    _, ext = os.path.splitext(file_name)
    if not ext:
        ext = '.wav'  # Extensión por defecto
    
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    blob.download_to_filename(temp_file.name)
    temp_file.close()
    
    return temp_file.name


def save_to_firestore(document_data, collection_name):
    db = get_firestore_client()
    timestamp = document_data.get('timestamp')
    document_id = timestamp.strftime("%Y%m%d_%H%M%S")
    
    db.collection(collection_name).document(document_id).set(document_data)
    return document_id


def save_detection_results(is_alarm, file_name, bucket_name, predicted_class, confidence_score, 
                          context_sounds, urgency_level=None, urgency_confidence=None, 
                          description=None, volume_level=None, repetition_pattern=None):
    timestamp = datetime.utcnow()
    document_id = timestamp.strftime("%Y%m%d_%H%M%S")
    db = get_firestore_client()
    
    # Documento base con datos comunes
    base_doc = {
        'timestamp': timestamp,
        'file': file_name,
        'bucket': bucket_name,
        'class': predicted_class,
        'confidence': float(confidence_score),
        'context_sounds': context_sounds
    }
    
    # Proceso eficiente en lotes
    batch = db.batch()
    
    if not is_alarm:
        # Es un sonido no-alarma
        base_doc['ignored'] = True
        batch.set(db.collection('sound_detections_ignored').document(document_id), base_doc)
    else:
        # Es una alarma
        # Añadir campos específicos de alarma
        base_doc.update({
            'urgency_level': urgency_level,
            'urgency_confidence': float(urgency_confidence),
            'description': description,
            'volume_level': float(volume_level),
            'repetition_pattern': repetition_pattern
        })
        
        # Determinar colección basada en urgencia
        if urgency_level == "ALTA":
            collection = 'sound_alerts_high'
        elif urgency_level == "MEDIA":
            collection = 'sound_alerts_medium'
        elif urgency_level == "BAJA":
            collection = 'sound_alerts_low'
        else:  # MONITOREO
            collection = 'sound_monitoring'
        
        # Agregar a colección específica
        batch.set(db.collection(collection).document(document_id), base_doc)
        
        # Agregar a colección general
        batch.set(db.collection('sound_detections_all').document(document_id), base_doc)
    
    # Ejecutar todas las operaciones en una sola transacción
    batch.commit()
    
    return document_id, description


def save_error(file_name='unknown', bucket_name='unknown', error_msg=''):
    timestamp = datetime.utcnow()
    document_id = timestamp.strftime("%Y%m%d_%H%M%S")
    
    error_doc = {
        'timestamp': timestamp,
        'file': file_name,
        'bucket': bucket_name,
        'error': error_msg
    }
    
    db = get_firestore_client()
    db.collection('sound_processing_errors').document(document_id).set(error_doc)


def process_audio(event, context):
    # Información del archivo
    bucket_name = event['bucket']
    file_name = event['name']
    
    try:
        # Cargar modelo (se mantiene en caché entre invocaciones)
        model, class_names = load_model()
        sound_analyzer = SoundAnalyzer()
        
        # Descargar archivo
        temp_file_path = download_file(bucket_name, file_name)
        
        # Cargar audio - usar sr=16000 directamente para evitar remuestreo
        waveform, sr = librosa.load(temp_file_path, sr=16000)
        os.unlink(temp_file_path)  # Eliminar archivo temporal lo antes posible
        
        if len(waveform) < sr * 0.5:  # Audio demasiado corto
            return
        
        # Convertir a tensor y calcular volumen
        waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
        volume_level = min(1.0, np.mean(np.abs(waveform)) / 0.05)
        
        # Realizar predicción
        scores, embeddings, spectrogram = predict(waveform_tensor)
        scores_np = scores.numpy()
        scores_np_mean = scores_np.mean(axis=0)
        
        # Obtener las 5 clases principales
        top_indices = np.argsort(scores_np_mean)[-5:][::-1]
        top_index = top_indices[0]
        predicted_class = class_names[top_index]
        confidence_score = scores_np_mean[top_index]
        
        # Sonidos de contexto
        context_sounds = [class_names[idx] for idx in top_indices]
        
        # Verificar si el sonido es una alarma
        is_alarm = sound_analyzer.is_alarm_sound(predicted_class, context_sounds, confidence_score)
        
        if not is_alarm:
            # Guardar como no-alarma y terminar
            save_detection_results(
                False, file_name, bucket_name, predicted_class, confidence_score, context_sounds
            )
            return
        
        # Si la clase principal no es una alarma reconocida, buscar la mejor alarma en el contexto
        if predicted_class not in SoundClassData.ALARM_CLASSES:
            best_alarm_class, best_alarm_confidence = sound_analyzer.find_highest_confidence_alarm(scores_np_mean, class_names)
            
            if best_alarm_class:
                predicted_class = best_alarm_class
                confidence_score = best_alarm_confidence
        
        # Analizar patrón de repetición
        repetition_pattern = sound_analyzer.analyze_repetition_pattern(scores_np, top_index)
        
        # Determinar nivel de urgencia
        urgency_level, urgency_confidence, description = sound_analyzer.get_urgency_level(
            predicted_class, volume_level, repetition_pattern, context_sounds
        )
        
        # Guardar resultados en Firestore
        _, description = save_detection_results(
            True, file_name, bucket_name, predicted_class, confidence_score,
            context_sounds, urgency_level, urgency_confidence, description, 
            volume_level, repetition_pattern
        )
        
        # Devolver la descripción
        return description
        
    except Exception as e:
        # Registro de errores
        error_msg = str(e)
        save_error(
            file_name if 'file_name' in locals() else 'unknown',
            bucket_name if 'bucket_name' in locals() else 'unknown',
            error_msg
        )
        raise