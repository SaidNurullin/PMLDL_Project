from typing import List
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from generate_music import generate
import os


app = FastAPI()

class GenerateMusicScheme(BaseModel):
    length: int
    instrument: List[str]

@app.post("/generate")
async def generate_music_endpoint(generate_music_scheme: GenerateMusicScheme):
    print(generate_music_scheme)
    midi_file = "generated_music.mid"
    generate(generate_music_scheme.length, generate_music_scheme.instrument, midi_file)
    if os.path.exists(midi_file):
        return FileResponse(midi_file, media_type="audio/midi", filename="generated_music.mid")
    else:
        return {"error": "File not found"}