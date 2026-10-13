import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from sentence_transformers import SentenceTransformer
from starlette.concurrency import run_in_threadpool

app = FastAPI(title="Card Vision Service")

# Loaded once at startup, not per request. First run downloads the model
# weights (~350MB) — that's a one-time cost. After that, loading is fast
# and encoding a single image takes a fraction of a second.
model = SentenceTransformer("clip-ViT-B-32")


@app.post("/embed")
async def embed_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read that as an image.")

    # model.encode() is synchronous, CPU-heavy work. Running it directly here
    # would block FastAPI's event loop, serializing every request even when
    # several come in at once. run_in_threadpool hands it to a background
    # thread instead, so concurrent embedding requests actually run in parallel.
    embedding = await run_in_threadpool(model.encode, image)
    return {"embedding": embedding.tolist()}


@app.get("/health")
def health():
    return {"status": "ok"}