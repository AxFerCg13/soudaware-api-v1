from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from services.audio_processor import AudioProcessor
from io import BytesIO
import logging

audio = APIRouter()
audio_processor = AudioProcessor(desired_sample_rate=16000)

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@audio.post("/audio")
async def process_audio(file: UploadFile = File(...)):
    try:
        logger.info(f"Iniciando procesamiento de archivo: {file.filename}")
        
        if not file.filename.lower().endswith(('.wav')):
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
            result = audio_processor.process_audio_file(BytesIO(audio_file))
        except Exception as e:
            error_msg = f"Error en AudioProcessor: {str(e)}"
            raise HTTPException(status_code=500, detail=error_msg)
        
        required_keys = ["class", "date", "confidence"]
        if not all(key in result for key in required_keys):
            error_msg = f"Estructura de resultado inválida. Esperado: {required_keys}, Obtenido: {result.keys()}"
            raise HTTPException(status_code=500, detail=error_msg)
        
        response_data = {
            "class": result["class"],
            "date": result["date"],
            "confidence": result["confidence"],
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error inesperado: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)