import logging
import time

from fastapi import FastAPI, File, HTTPException
from fastapi.responses import JSONResponse
from processors import process_text, process_video

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.post(
    "/embed/video",
    description="Send .mp4(string bytes) file to get embedding",
    response_description="The processed video file(string bytes) embedidng.",
    response_class=JSONResponse,
)
async def embed_video(video_file: bytes = File()):
    start_time = time.time()
    try:
        golos = "temp_papka/video.mp4"
        with open(golos, "wb") as buffer:
            buffer.write(video_file)
        logger.info(f"video saved: {time.time() - start_time}")
        video_emb = process_video(golos)
        logger.info(f"get embs: {time.time() - start_time}")
        return JSONResponse(
            content={"embedding": video_emb.tolist()[0]},
            headers={
                "response_time": f"{time.time() - start_time}",
            },
        )
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/embed/text")
async def embed_text(prompt: str):
    start_time = time.time()
    try:
        text_emb = process_text(prompt)
        headers = {
            "response_time": f"{time.time() - start_time}",
        }
        return JSONResponse(content={"embedding": text_emb.tolist()[0]}, headers=headers)
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn  # TOLKO DLAY TESTOV NE BOLEE :D

    uvicorn.run(app, host="0.0.0.0", port=48521)  # noqa: S104
