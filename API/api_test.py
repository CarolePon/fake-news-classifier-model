from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from FakeNews_packages import model_ml  # Assuming this includes your models
from fastapi.middleware.cors import CORSMiddleware
from FakeNews_packages import preprocessing2
import random

# Initialize FastAPI app
app = FastAPI()

# Allow all CORS origins for development purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    title_input_string: str = None  # Optional
    text_input_string: str = None  # Optional

# Define the models (replace with actual model loading logic)
model_text_only = model_ml.model_text_only
model_title_only = model_ml.model_title_only
model_both = model_ml.model_both
model_vote = model_ml.model_vote


@app.get('/')
def index():
    return {'ok': True}


@app.post('/predict')
def predict(request: PredictionRequest):


    """
    Prediction endpoint for fake news classification.
    """

    # Preprocess inputs
    title = request.title_input_string.strip() if request.title_input_string else None
    text = request.text_input_string.strip() if request.text_input_string else None

    # Check if at least one input is provided
    if not title and not text:
        raise HTTPException(status_code=400, detail="At least one of 'title_input_string' or 'text_input_string' must be provided.")

    # Initialize response structure
    response = {"title_model_result": None, "text_model_result": None, "combined_model_result": None, "final_result": None}

    # Case 1: Both title and text are provided
    if title and text:
        # Preprocess inputs
        title_preprocessed = preprocessing2.preproc_predict(title)
        text_preprocessed = preprocessing2.preproc_predict(text)

        # Get predictions from all models UNCOMMENT ONCE WE HAVE MODELS
        title_result = model_title_only()
        text_result = model_text_only()
        combined_result = model_both()
        # title_result = model_title_only.predict([title_preprocessed])[0]
        # text_result = model_text_only.predict([text_preprocessed])[0]
        # combined_result = model_both.predict([[title_preprocessed, text_preprocessed]])[0]

        # Majority voting mechanism
        final_result = random.choice([title_result, text_result, combined_result])
 #  [title_result, text_result, combined_result]) UNCOMMENT WHEN WE HAVE MODELS

        # Update response UNCOMMENT WHEN WE HAVE MODELS
        response.update({
            "title_model_result": title_result,
            "text_model_result": text_result,
            "combined_model_result": combined_result,
            "final_result": final_result
        })

    # Case 2: Only title is provided
    elif title:
        title_preprocessed = preprocessing2.preproc_predict(title)

        title_result = model_title_only()  #.predict([title_preprocessed])[0] UNCOMMENT W+OCNE WE HAVE MODELS
        #response["final_result"] = title_result UNCOMMENT ONCE WE HAVE MODELS
        response = {"title_model_result": title_result, "final_result": title_result}

    # Case 3: Only text is provided
    elif text:
        text_preprocessed = preprocessing2.preproc_predict(text)
        text_result = model_text_only()  #.predict([text_preprocessed])[0] UNCOMMENT W+OCNE WE HAVE MODELS
        response = {"text_model_result": text_result, "final_result": text_result} #DEL ONCE WE HAVE THE MODELS
        #response["final_result"] = text_result UNCOMMENT ONCE WE HAVE MDOELS

    # Return the response
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_test:app", host="0.0.0.0", port=8000)
