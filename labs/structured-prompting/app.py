import os
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class ExtractedEntity(BaseModel):
    name: str = Field(description="Name of person or organization")
    category: str = Field(description="Classification: 'Person', 'Organization', or 'Location'")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")

class ExtractionSchema(BaseModel):
    entities: list[ExtractedEntity]

def extract_structured_data(text: str) -> ExtractionSchema:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract named entities into the specified JSON format strictly."},
            {"role": "user", "content": text}
        ],
        response_format=ExtractionSchema
    )
    return completion.choices[0].message.parsed

if __name__ == "__main__":
    sample_text = "OpenAI announced a partnership with Apple to integrate ChatGPT into iOS in San Francisco."
    data = extract_structured_data(sample_text)
    for entity in data.entities:
        print(f"[{entity.category}] {entity.name} (Confidence: {entity.confidence})")
