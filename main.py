from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from endpoints.predict import router as predict_router

app = FastAPI(title="Lip Reading API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Lip Reading API"}
