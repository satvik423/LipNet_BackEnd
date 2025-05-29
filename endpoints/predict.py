from fastapi import APIRouter, UploadFile, File
from services.lipreader import run_prediction
import tempfile

router = APIRouter()

@router.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    result_text = run_prediction(tmp_path)
    return {"prediction": result_text}
