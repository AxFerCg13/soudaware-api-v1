from fastapi import FastAPI
from routes.audio import audio

app = FastAPI()

app.include_router(audio)