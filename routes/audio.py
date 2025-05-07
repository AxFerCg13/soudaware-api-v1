from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

audio = APIRouter()

@audio.post("/audio")
async def upload_file(file: UploadFile = File(...)):
    try:
        print(file)
        with open(file.filename, "wb") as audioWav:
            content = await file.read()
            print(content)
        
        return JSONResponse(content={
            "success": True,
        }, status_code=200)

    except FileNotFoundError:
        return JSONResponse(content=
                            {
                                "success": False,
                            }, status_code=404)
    return "Hello world"