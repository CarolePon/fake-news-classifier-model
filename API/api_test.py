from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def index():
    return {'ok': True}

@app.get('/predict')
def predict():
    return {"we cannot tell yet, we are in implementation phase to learn patience": 'La patience et le temps font plus que force ou rage'}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_test:app", host="0.0.0.0", port=8000)
