"""
Finance-DeepSeek FastAPI service layer.

Exposes REST endpoints for:
  - Health checks
  - RAG retrieval
  - End-to-end reasoning generation
  - Standalone chain parsing
  - Standalone chain validation
  - Index management
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import (
    HealthResponse,
    RetrieveRequest,
    RetrieveResponse,
    ChunkResult,
    GenerateRequest,
    GenerateResponse,
    ParseRequest,
    ParseResponse,
    ValidateRequest,
    ValidateResponse,
    IndexBuildRequest,
    IndexBuildResponse,
    IndexStatusResponse,
)
from api.service import FinanceDeepSeekService

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan: lazy service init
# ---------------------------------------------------------------------------

service: FinanceDeepSeekService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = FinanceDeepSeekService()
    logger.info("FinanceDeepSeekService initialized.")
    yield
    logger.info("Shutting down FinanceDeepSeekService.")


# ---------------------------------------------------------------------------
# Helper: ensure service is ready (fallback for test environments)
# ---------------------------------------------------------------------------

def _get_service() -> FinanceDeepSeekService:
    global service
    if service is None:
        service = FinanceDeepSeekService()
    return service


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Finance-DeepSeek API",
    description="FastAPI service for financial reasoning with DeepSeek-R1-Distill-Qwen",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static test console
# ---------------------------------------------------------------------------

# Mount the interactive HTML test console at /test
TEST_HTML_PATH = PROJECT_ROOT / "test_api.html"
if TEST_HTML_PATH.exists():
    app.mount("/test", StaticFiles(directory=str(PROJECT_ROOT), html=True), name="test")
    logger.info(f"Test console mounted at /test (source: {TEST_HTML_PATH})")
else:
    logger.warning(f"test_api.html not found at {TEST_HTML_PATH}; /test endpoint unavailable")


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the test console."""
    return RedirectResponse(url="/test/test_api.html")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        model_loaded=_get_service().model_loaded,
        index_loaded=_get_service().index_loaded,
    )


# ---------------------------------------------------------------------------
# RAG Retrieval
# ---------------------------------------------------------------------------

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    """Retrieve relevant chunks from the FAISS index."""
    try:
        chunks, context = _get_service().retrieve(
            query=req.query,
            top_k=req.top_k,
            rerank_top_k=req.rerank_top_k,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Retrieve failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    chunk_results = [
        ChunkResult(
            text=c.get("text", ""),
            source_url=c.get("source_url", "未知来源"),
            category=c.get("category"),
            title=c.get("title"),
            score=c.get("score", 0.0),
        )
        for c in chunks
    ]

    return RetrieveResponse(
        query=req.query,
        chunks=chunk_results,
        context=context,
    )


# ---------------------------------------------------------------------------
# Generation (end-to-end)
# ---------------------------------------------------------------------------

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """
    Full pipeline: optional RAG retrieval, LLM generation, chain parsing,
    and consistency validation.
    """
    try:
        result = _get_service().answer(
            query=req.query,
            use_rag=req.use_rag,
            temperature=req.temperature,
            max_new_tokens=req.max_new_tokens,
            top_p=req.top_p,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return GenerateResponse(
        query=result["query"],
        reasoning_steps=result["reasoning_steps"],
        final_answer=result["final_answer"],
        confidence=result["confidence"],
        fallback_triggered=result["fallback_triggered"],
        fallback_reason=result["fallback_reason"],
        sources=result["sources"],
        raw_output=result["raw_output"],
        validated=result["validated"],
        validation_message=result["validation_message"],
    )


# ---------------------------------------------------------------------------
# Standalone Chain Parsing
# ---------------------------------------------------------------------------

@app.post("/parse", response_model=ParseResponse)
async def parse(req: ParseRequest):
    """Parse raw model output into structured reasoning steps and answer."""
    try:
        parsed = _get_service().parse_raw_output(req.raw_text, sources=req.sources)
    except Exception as e:
        logger.exception("Parse failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return ParseResponse(
        reasoning_steps=parsed["reasoning_steps"],
        final_answer=parsed["final_answer"],
        confidence=parsed["confidence"],
        fallback_triggered=parsed["fallback_triggered"],
        fallback_reason=parsed.get("fallback_reason", ""),
        sources=parsed.get("sources", []),
    )


# ---------------------------------------------------------------------------
# Standalone Validation
# ---------------------------------------------------------------------------

@app.post("/validate", response_model=ValidateResponse)
async def validate(req: ValidateRequest):
    """Validate consistency between reasoning chain and final answer."""
    try:
        is_consistent, reason = _get_service().validate_chain(req.reasoning_steps, req.final_answer)
    except Exception as e:
        logger.exception("Validation failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return ValidateResponse(is_consistent=is_consistent, reason=reason)


# ---------------------------------------------------------------------------
# Index Management
# ---------------------------------------------------------------------------

@app.get("/index/status", response_model=IndexStatusResponse)
async def index_status():
    """Get current FAISS index status."""
    status_info = _get_service().get_index_status()
    return IndexStatusResponse(**status_info)


@app.post("/index/build", response_model=IndexBuildResponse)
async def index_build(req: IndexBuildRequest):
    """Build or rebuild the FAISS index from a corpus directory."""
    try:
        result = _get_service().build_index(corpus_dir=req.corpus_dir)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Index build failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return IndexBuildResponse(**result)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    cfg_path = PROJECT_ROOT / "config.yaml"
    if cfg_path.exists():
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        api_cfg = cfg.get("api", {})
        host = api_cfg.get("host", "0.0.0.0")
        port = api_cfg.get("port", 8000)
    else:
        host, port = "0.0.0.0", 8000

    uvicorn.run("api.main:app", host=host, port=port, reload=False)
