from typing import List, Optional, Union

from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    model: str = Field(..., description="ID of the embedding model to use")
    input: Union[str, List[str]] = Field(..., description="Input text(s) to embed")
    encoding_format: Optional[str] = Field("float", description="Encoding format: 'float' or 'base64'")


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage
