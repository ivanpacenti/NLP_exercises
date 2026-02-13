from fastapi import FastAPI, UploadFile, File, HTTPException
import fitz
import re

app = FastAPI()

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

@app.post("/v1/extract-sentences")
async def extract_sentences(pdf_file: UploadFile = File(...)):
    data = await pdf_file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open PDF: {e}")

    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    doc.close()

    text = "\n".join(parts)

    # basic cleanup
    text = text.replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text)

    sentences = SENTENCE_RE.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    return {"sentences": sentences}