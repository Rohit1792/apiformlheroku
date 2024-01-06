from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import uvicorn
from pyngrok import ngrok
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio

app = FastAPI()


origins = ["*"]
app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])


class model_input(BaseModel):
    Pregnancies : int
    Glucose : int
    Blood_Pressure : int
    Skin_Thickness : int
    Insulin : int
    BMI : float
    DPF : float
    Age :  int

# Loading the Model

diabetes_model = pickle.load(open("diabetes_model.sav", "rb"))

@app.post("/diabetes_prediction")
def diabetes_pred(input_parameters : model_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    preg = input_dictionary["Pregnancies"]
    glu = input_dictionary["Glucose"]
    bp = input_dictionary["Blood_Pressure"]
    skin = input_dictionary["Skin_Thickness"]
    ins = input_dictionary["Insulin"]
    bmi = input_dictionary["BMI"]
    dpf = input_dictionary["DPF"]
    age = input_dictionary["Age"]


    input_list = [preg, glu, bp, skin, ins, bmi, dpf, age]

    prediction = diabetes_model.predict([input_list])

    if prediction[0] == 0:
        return "The Person is not Diabetic."
    else:
        return "The Person is diabetic"
    
ngrok_tunnel = ngrok.connect(8000)
print("Public URL: ", ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)