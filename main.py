from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import torch

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    url: str

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma(persist_directory="./chroma_db_alt", embedding_function=embedding_function)

def similarity_search(query, k=5):
    results = vectordb.similarity_search(query, k=k)
    return results

model_id = "meta-llama/Meta-Llama-Guard-2-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device)

def generate_rag_answer(question, k=3):
    results = similarity_search(question, k)
    
    context = "\n\n".join([f"Source {i+1}: {result.page_content}" for i, result in enumerate(results)])
    
    prompt = (
        f"You are an expert on VA disability benefits. You will be provided with context information from several sources.\n"
        f"Use only the provided information to answer the question accurately and concisely. Do not use any external knowledge. "
        f"Base your answer strictly on the context provided.\n\n"
        f"Please follow these instructions carefully to provide an accurate answer:\n"
        f"1. Carefully read each paragraph of the provided content.\n"
        f"2. Identify if the paragraph contains relevant information to answer the question. If a paragraph does not provide relevant information, move to the next paragraph.\n"
        f"3. When you find relevant information, use it to construct your answer. Include as much evidence as possible from the context to support your answer, even if an answer has already been started.\n"
        f"4. Ensure your answer is accurate, concise, and based solely on the provided context.\n\n"
        f"Context:\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=500, pad_token_id=tokenizer.pad_token_id, temperature=0)
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    answer = generated_text.split("Answer:")[1].strip() if "Answer:" in generated_text else generated_text.strip()
    
    first_url = results[0].metadata['source_url'] if results else "No URL found"
    
    return answer, first_url

@app.post("/generate-answer", response_model=AnswerResponse)
def generate_answer(request: QuestionRequest):
    try:
        answer, url = generate_rag_answer(request.question)
        return AnswerResponse(answer=answer, url=url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

