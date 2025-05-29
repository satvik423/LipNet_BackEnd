from fastapi import FastAPI
from endpoints.predict import router as predict_router

app = FastAPI(title="Lip Reading API")

app.include_router(predict_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Lip Reading API"}
