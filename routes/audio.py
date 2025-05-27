from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from services.audio_processor import AudioProcessor
from io import BytesIO
from googletrans import Translator
import logging
import numpy as np

audio = APIRouter()
audio_processor = AudioProcessor(desired_sample_rate=16000)

# Configura logging
logging.basicConfig(level=logging.INFO)

@audio.post("/audio")
async def process_audio(file: UploadFile = File(...)):
    try:
        
        # Validaciones básicas
        if not file.filename.lower().endswith(('.wav', '.mp3')):
            raise HTTPException(status_code=400, detail="Formato de archivo no soportado")
            
        audio_content = await file.read()
        if len(audio_content) == 0:
            raise HTTPException(status_code=400, detail="Archivo vacío recibido")
        
        # Procesamiento
        try:
            result = await audio_processor.process_audio_file(BytesIO(audio_content))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error procesando audio: {str(e)}")
        
        # Validar y formatear respuesta
        required_keys = ["class", "date", "confidence", "is_alarm"]
        if not all(key in result for key in required_keys):
            raise HTTPException(status_code=500, detail="Estructura de resultado inválida")
        
        # Convertir numpy/tensorflow types a Python nativo
        processed_result = {
            k: v.item() if hasattr(v, 'item') else v.tolist() if isinstance(v, (np.generic, np.ndarray)) else v
            for k, v in result.items()
        }
        
        # Construir respuesta
        response = {
            "is_alarm": processed_result["is_alarm"],
            "classMessage": processed_result["class"],
            "confidence": processed_result["confidence"],
            "date": processed_result["date"],
            "context_sounds": processed_result.get("context_sounds", []),
            "firestore_id": processed_result.get("firestore_id"),
            "firestore_error": processed_result.get("firestore_error")
        }
        
        if processed_result["is_alarm"]:
            response.update({
                "urgency_level": processed_result.get("urgency_level"),
                "description": processed_result.get("description"),
                "recommendations": processed_result.get("recommendations", [])
            })
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error interno del servidor")