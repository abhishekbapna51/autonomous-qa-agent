from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path

from .models import (
    BuildKBResponse,
    GenerateTestCasesRequest,
    GenerateTestCasesResponse,
    GenerateScriptRequest,
    GenerateScriptResponse,
)
from .config import settings
from .deps import (
    ingestion_service,
    test_generation_service,
    script_generation_service,
)
from .utils.logging_utils import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Autonomous QA Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/api/build_kb", response_model=BuildKBResponse)
async def build_kb(files: List[UploadFile] = File(...)):
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for f in files:
        dest = upload_dir / f.filename
        logger.info(f"Saving uploaded file: {dest}")
        with dest.open("wb") as out:
            out.write(await f.read())
        saved_paths.append(dest)

    num_chunks = ingestion_service.build_kb_from_paths(saved_paths)
    return BuildKBResponse(
        status="success",
        num_documents=len(saved_paths),
        num_chunks=num_chunks,
        message="Knowledge base built successfully.",
    )


@app.post("/api/generate_test_cases", response_model=GenerateTestCasesResponse)
async def generate_test_cases(payload: GenerateTestCasesRequest):
    test_cases = test_generation_service.generate_test_cases(payload.query)
    return GenerateTestCasesResponse(test_cases=test_cases)


@app.post("/api/generate_selenium_script", response_model=GenerateScriptResponse)
async def generate_selenium_script(payload: GenerateScriptRequest):
    script = script_generation_service.generate_script(payload.test_case)
    return GenerateScriptResponse(script=script)
