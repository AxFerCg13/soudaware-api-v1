from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from services.audio_processor import AudioProcessor
from io import BytesIO

audio = APIRouter()
audio_processor = AudioProcessor(desired_sample_rate=16000)

@audio.post("/audio")
async def process_audio(file: UploadFile = File(...)):
    audio_file = await file.read()
    
    result = audio_processor.process_audio_file(BytesIO(audio_file))
    
    return JSONResponse(content={
        "class": result["class"],
        "confidence": result["confidence"],
    })