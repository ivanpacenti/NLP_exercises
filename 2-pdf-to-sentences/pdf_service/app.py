from fastapi import FastAPI, UploadFile, File, HTTPException
import fitz
import re

app = FastAPI()

# Keep letters (EN + DK) and apostrophes inside words, remove other special characters
KEEP_RE = re.compile(r"[^A-Za-zÆØÅæøå'’]+")

def clean_token(t: str) -> str:
    # Remove soft hyphen if present (-)
    t = t.replace("\u00ad", "")
    # Drop other special chars (keep letters + apostrophes)
    t = KEEP_RE.sub("", t)
    return t

@app.post("/v1/pdf-to-words")
async def pdf_to_words(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open PDF: {e}")

    parts = []
    try:
        # Extract text from each page
        for page in doc:
            parts.append(page.get_text("text"))
    finally:
        doc.close()
    #join all parts into one string
    text = "\n".join(parts)

    # Basic whitespace tokenization
    toks = text.split()

    # Merge hyphen-at-end with next token
    # Example:
    # ["ex-", "traction"]  ->  ["extraction"]
    # it scans the token list manually using an index.
    merged = []
    i = 0

    while i < len(toks):
        t = toks[i]

        # If the current token ends with a hyphen
        # and there is a next token available,
        # they are joined together.
        if t.endswith("-") and i + 1 < len(toks):
            # Remove the trailing hyphen from current token
            # and concatenate it with the next token.
            merged.append(t[:-1] + toks[i + 1])

            # Skip the next token because already merged.
            i += 2
        else:
            # Otherwise, just keep the token as it is.
            merged.append(t)
            i += 1

    # 3) Clean tokens and drop empties
    out = []

    for t in merged:
        # clean_token() should remove special chars, strip whitespaces and remove soft hyphens
        ct = clean_token(t)

        # Only keep non-empty tokens
        if ct:
            out.append(ct)

    # Return total number of cleaned words and first 5000 words
    return {"n_words": len(out), "words": out[:5000]}