from fastapi import FastAPI

app = FastAPI(title="CRISI API v2")

@app.get("/health")
def health():
    return {"status": "ok"}
