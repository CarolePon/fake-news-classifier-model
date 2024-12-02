from fastapi import FastAPI
import requests
from FakeNews_packages import model_ml

from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# a changer
app.state.model = model_ml

@app.get('/')
def index():
    return {'ok': True}

@app.get('/predict')
def predict():
    return {"we cannot tell yet, we are in implementation phase to learn patience": 'La patience et le temps font plus que force ou rage'}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_test:app", host="0.0.0.0", port=8000)
