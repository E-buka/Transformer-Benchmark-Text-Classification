from fastapi import FastAPI, HTTPException
from pydantic import BaseModel 
from contextlib import asynccontextmanager 
from fastapi.responses import Response 
from src.inference import load_model 

@asynccontextmanager 
async def lifespan(app: FastAPI):
    try: 
        app.state.classifier = load_model()
        print("Application started succesfully")
        yield 
    except Exception as e: 
        raise RuntimeError(f"App loading failed: {e}")
    finally: 
        print("Application shutting down") 
        
app = FastAPI(lifespan=lifespan, title="Text Classification")


class UserInput(BaseModel): 
    text: str  
    
@app.get("/") 
def root():
    return {"status": "ok",
            "message": "Use POST/predict"}
    
@app.get("/health")
def health():
    return {"status": "ready to accept user input"}

@app.get("/favicon.ico")
async def get_favicon():
    return Response(status_code=200) 

@app.post("/predict")
async def predict(payload: UserInput): 
    try:
        output = app.state.classifier(payload.text)
        return output 
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))
    