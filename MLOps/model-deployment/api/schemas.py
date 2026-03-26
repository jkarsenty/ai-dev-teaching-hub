from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

class ImageResponse(BaseModel):
    digit: int
    confidence: float