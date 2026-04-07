from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from video_processor import process_video
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="."), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for development
    allow_methods=["*"],
    allow_headers=["*"],
)



import asyncio
from functools import partial
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Run blocking function in a thread so the server doesn't freeze
    loop = asyncio.get_event_loop()
    video_path, csv_path = await loop.run_in_executor(None, process_video, file_path)
    return {
        "message": "Processing complete",
        "video_url": "http://127.0.0.1:8000/video",
        "csv_url": "http://127.0.0.1:8000/csv"
    }


from fastapi.responses import FileResponse

from fastapi import Request
from fastapi.responses import StreamingResponse
import os
@app.get("/video")
def get_video(request: Request):
    video_path = "saved_tracked_video.mp4"
    file_size = os.path.getsize(video_path)
    range_header = request.headers.get("Range")
    if range_header:
        # Parse the range: "bytes=start-end"
        range_val = range_header.strip().split("=")[1]
        start, end = range_val.split("-")
        start = int(start)
        end = int(end) if end else file_size - 1
        chunk_size = end - start + 1
        def stream():
            with open(video_path, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    read_size = min(remaining, 65536)
                    data = f.read(read_size)
                    if not data:
                        break
                    yield data
                    remaining -= len(data)
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
        }
        return StreamingResponse(stream(), status_code=206, headers=headers, media_type="video/mp4")
    # Full file if no Range header
    def stream_full():
        with open(video_path, "rb") as f:
            yield from iter(lambda: f.read(65536), b"")
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
    }
    return StreamingResponse(stream_full(), headers=headers, media_type="video/mp4")

@app.get("/csv")
def get_csv():
    return FileResponse(
        path="traffic_log.csv",
        media_type="text/csv",
        filename="traffic_log.csv"
    )

@app.get("/")
def home():
    return FileResponse("index.html")