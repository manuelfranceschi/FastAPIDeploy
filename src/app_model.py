# Librerias
from fastapi import FastAPI, requests, HTTPException
from pydantic import BaseModel
import sqlite3
import uvicorn
import os
import pickle
import pandas as pd
from typing import List, Tuple
from typing import Optional

app = FastAPI()

class DataPredict(BaseModel):
    data: List[list[float, float, float]]

class DataIngest(BaseModel):
    data: List[list[float, float, float]]

@app.get("/")
async def hello():
    return "Bienvenido a mi API :)"

# 0. Mostrar registros
@app.get("/database")
async def show_database():
    try:
        conn = sqlite3.connect("./data/advertising.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM advertising")
        results = cursor.fetchall()
        conn.close()
        return dict({"results":results})
    except Exception as e:
        raise HTTPException(status_code=500, detail={str(e)})

# 1. Endpoint de predicción
@app.get("/predict")
async def predict_sales(data: DataPredict):
    model = pickle.load(open('./model/advertising_model.pkl','rb'))
    if not data:
        raise HTTPException(status_code=400, detail = "No se han proporcionado datos")
    try:
        if len(data.data[0]) == 3:
            prediction = model.predict(data.data)[0]
            return {"prediction":prediction}
    except Exception as e: 
            raise HTTPException(status_code=500, detail=str(e))
    
# 2. Endpoint de ingesta de datos
@app.post("/ingest")
async def ingest_data(data: DataIngest):
    if not data: 
        raise HTTPException(status_code=400, detail = "No se han proporcionado datos nuevos.")
    try:
        conn = sqlite3.connect("./data/advertising.db")
        cursor = conn.cursor()
        if len(data.data[0]) == 4:
            cursor.execute(
                    """
                        INSERT INTO advertising (tv, radio, newspaper, sales)
                        VALUES (?, ?, ?, ?)
                    """,
                    (data.data[0][0], data.data[0][1], data.data[0][2], data.data[0][3])
                )
            conn.commit()
            conn.close()
            return {"message": "Datos ingresados correctamente"}
    except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# 2. Endpoint de reentramiento del modelo
@app.post("/retrain")
async def retrain_model():
    try:
        conn = sqlite3.connect("./data/advertising.db")
        data = pd.read_sql_query("SELECT * FROM advertising", conn)
        X = data[["TV", "radio", "newspaper"]]
        y = data["sales"]
        model = pickle.load(open('./model/advertising_model.pkl','rb'))
        model.fit(X, y)
        with open("./model/advertising_model.pkl", "wb") as file:
                pickle.dump(model, file)
        return {'message': 'Modelo reentrenado correctamente.'}
    except Exception as e: 
        raise HTTPException(status_code=500, detail=f"Error al reentrenar el modelo: {str(e)}")

# Carga de FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
