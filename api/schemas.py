"""
Pydantic schemas for the Finance-DeepSeek FastAPI service.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health & Status
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether the LLM is loaded")
    index_loaded: bool = Field(..., description="Whether the FAISS index is loaded")


# ---------------------------------------------------------------------------
# RAG Retrieval
# ---------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query string")
    top_k: int = Field(5, ge=1, le=50, description="Number of chunks to retrieve")
    rerank_top_k: Optional[int] = Field(None, ge=1, le=50, description="Post-rerank top-k")


class ChunkResult(BaseModel):
    text: str = Field(..., description="Chunk text content")
    source_url: str = Field("未知来源", description="Source URL")
    category: Optional[str] = Field(None, description="Document category")
    title: Optional[str] = Field(None, description="Document title")
    score: float = Field(..., description="Similarity score")


class RetrieveResponse(BaseModel):
    query: str = Field(..., description="Original query")
    chunks: List[ChunkResult] = Field(..., description="Retrieved chunks")
    context: str = Field(..., description="Formatted context string for prompt injection")


# ---------------------------------------------------------------------------
# Reasoning Generation
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User financial question")
    context: Optional[str] = Field(None, description="Optional external context")
    use_rag: bool = Field(True, description="Whether to retrieve context from RAG")
    temperature: float = Field(0.6, ge=0.0, le=2.0, description="Sampling temperature")
    max_new_tokens: int = Field(1024, ge=1, le=4096, description="Max tokens to generate")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")


class ReasoningStep(BaseModel):
    step_number: int = Field(..., description="Step index (1-based)")
    content: str = Field(..., description="Step content")


class GenerateResponse(BaseModel):
    query: str = Field(..., description="Original query")
    reasoning_steps: List[str] = Field(..., description="Parsed reasoning steps")
    final_answer: str = Field(..., description="Final extracted answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    fallback_triggered: bool = Field(..., description="Whether fallback parsing was used")
    fallback_reason: str = Field("", description="Reason for fallback if triggered")
    sources: List[str] = Field(default_factory=list, description="Retrieved source URLs")
    raw_output: str = Field(..., description="Raw model output")
    validated: bool = Field(..., description="Whether chain validation passed")
    validation_message: str = Field("", description="Validation result message")


# ---------------------------------------------------------------------------
# Chain Parsing (standalone)
# ---------------------------------------------------------------------------

class ParseRequest(BaseModel):
    raw_text: str = Field(..., min_length=1, description="Raw model output to parse")
    sources: Optional[List[str]] = Field(None, description="Optional source URLs")


class ParseResponse(BaseModel):
    reasoning_steps: List[str] = Field(..., description="Parsed reasoning steps")
    final_answer: str = Field(..., description="Final extracted answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    fallback_triggered: bool = Field(..., description="Whether fallback was triggered")
    fallback_reason: str = Field("", description="Reason for fallback")
    sources: List[str] = Field(default_factory=list, description="Source URLs")


# ---------------------------------------------------------------------------
# Validation (standalone)
# ---------------------------------------------------------------------------

class ValidateRequest(BaseModel):
    reasoning_steps: List[str] = Field(..., description="List of reasoning steps")
    final_answer: str = Field(..., description="Final answer string")


class ValidateResponse(BaseModel):
    is_consistent: bool = Field(..., description="Whether chain and answer are consistent")
    reason: str = Field(..., description="Validation message")


# ---------------------------------------------------------------------------
# Index Management
# ---------------------------------------------------------------------------

class IndexBuildRequest(BaseModel):
    corpus_dir: Optional[str] = Field(None, description="Path to corpus directory")


class IndexBuildResponse(BaseModel):
    success: bool = Field(..., description="Whether build succeeded")
    vectors_added: int = Field(..., description="Number of vectors indexed")
    index_path: str = Field(..., description="Path to saved index")
    metadata_path: str = Field(..., description="Path to saved metadata")


class IndexStatusResponse(BaseModel):
    index_loaded: bool = Field(..., description="Whether index is loaded")
    total_vectors: int = Field(0, description="Total vectors in index")
    index_path: Optional[str] = Field(None, description="Index file path")
    metadata_path: Optional[str] = Field(None, description="Metadata file path")
