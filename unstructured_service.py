# unstructured_service.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
import os

app = FastAPI()
UPLOAD_DIR = "/tmp/unstructured_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/chunk")
async def chunk_document(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Process the file with unstructured
        elements = partition(filename=file_path)
        chunks = chunk_by_title(elements)

        # Convert chunk objects to a simple list of strings
        text_chunks = [chunk.text for chunk in chunks]

        return {"chunks": text_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)