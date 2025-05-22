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
logger = logging.getLogger(__name__)

@audio.post("/audio")
async def process_audio(file: UploadFile = File(...)):
    try:
        logger.info(f"Iniciando procesamiento de archivo: {file.filename}")
        
        if not file.filename.lower().endswith(('.wav', '.mp3')):
            error_msg = "Formato de archivo no soportado"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
            
        audio_file = await file.read()
        
        if len(audio_file) == 0:
            error_msg = "Archivo vacío recibido"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        logger.info(f"Archivo recibido - Tamaño: {len(audio_file)} bytes")
        
        # Procesamiento del audio
        try:
            result = await audio_processor.process_audio_file(BytesIO(audio_file))
        except Exception as e:
            error_msg = f"Error en AudioProcessor: {str(e)}"
            raise HTTPException(status_code=500, detail=error_msg)
        
        required_keys = ["class", "date", "confidence", "is_alarm"]
        if not all(key in result for key in required_keys):
            error_msg = f"Estructura de resultado inválida. Esperado: {required_keys}, Obtenido: {result.keys()}"
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Convertir valores numéricos a tipos nativos de Python
        def convert_value(v):
            if hasattr(v, 'item'):  # Para numpy/tensorflow types
                return v.item()
            elif isinstance(v, (np.generic, np.ndarray)):  # Para numpy arrays/scalars
                return v.tolist()
            return v
        
        # Aplicar la conversión a todos los valores numéricos
        processed_result = {k: convert_value(v) for k, v in result.items()}
        
        if processed_result["confidence"] < 0.19:
            processed_result["class"] = "Loud Sound"

        # Construir respuesta base
        response_data = {
            "is_alarm": processed_result["is_alarm"],
            "classMessage": processed_result["class"],
            "confidence": processed_result["confidence"],
            "context_sounds": processed_result.get("context_sounds", []),
            "date": processed_result["date"],
        }

        # Solo incluir campos avanzados si es una alarma
        if processed_result["is_alarm"]:
            response_data.update({
                "urgency_level": processed_result.get("urgency_level"),
                "description": processed_result.get("description"),
                "volume_level": processed_result.get("volume_level"),
                "repetition_pattern": processed_result.get("repetition_pattern"),
            })
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error inesperado: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)