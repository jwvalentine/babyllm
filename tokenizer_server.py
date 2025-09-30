from fastapi import FastAPI
from pydantic import BaseModel
from tokenizers import Tokenizer

app = FastAPI()
tokenizer = Tokenizer.from_file("models/all-MiniLM-L6-v2/tokenizer.json")

class TextInput(BaseModel):
    text: str

@app.post("/tokenize")
def tokenize(req: TextInput):
    encoding = tokenizer.encode(req.text)
    return {
        "ids": encoding.ids,
        "attention_mask": encoding.attention_mask
    }
