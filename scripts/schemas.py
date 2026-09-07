"""Pydantic request/response models for the public API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Client-generated id used to group a conversation.")
    message: str = Field(..., min_length=1, description="The user's message.")


class ChatResponse(BaseModel):
    session_id: str
    response: str
