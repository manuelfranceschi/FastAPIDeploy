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

# Inicialización FastAPI y modelo
app = FastAPI()
model = pickle.load(open('../model/advertising_model.pkl','rb'))

# CSV a db
df = pd.read_csv("../data/Advertising.csv", index_col=0)
conn = sqlite3.connect("../data/advertising.db") # Si no existe esta db, la crea. Es más que nada para que pueda hacer lo siguiente:
cursor = conn.cursor()

#Gracias a esta función el id será autoincremental
cursor.execute("""
CREATE TABLE IF NOT EXISTS advertising ( 
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tv REAL,
    radio REAL,
    newspaper REAL,
    sales REAL
)
""")
df.to_sql("advertising", conn, if_exists="replace", index=True) #Finalmente, pasamos a database.

class DataPredict(BaseModel):
    data: List[list[float, float, float]]

class DataIngest(BaseModel):
    data: List[list[float, float, float]]

# Mostrar un saludo 
@app.get("/")
async def hello():
    print(conn)
    return "Bienvenido a mi API :)"

# 0. Mostrar registros
@app.get("/database")
async def show_database():
    try:
        cursor.execute("SELECT * FROM advertising")
        results = cursor.fetchall()
        return dict({"results":results})
    except Exception as e:
        raise HTTPException(status_code=500, detail={str(e)})

# 1. Endpoint de predicción
@app.get("/predict")
async def predict_sales(data: DataPredict):
    if len(data.data[0]) == 3:
        try:
            prediction = model.predict(data.data)[0]
            return {"prediction":prediction}
        except Exception as e: 
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail=f"Tienes que pasar 3 datos. ej: [[10.3, 300.5, 150.70]]. {str(e)}")
    
# 2. Endpoint de ingesta de datos
@app.post("/ingest")
async def ingest_data(data: DataIngest):
    if len(data.data[0]) == 4:
        try:
            cursor.execute(
                    """
                        INSERT INTO advertising (tv, radio, newspaper, sales)
                        VALUES (?, ?, ?, ?)
                    """,
                    (data.data[0][0], data.data[0][1], data.data[0][2], data.data[0][3])
                )
            conn.commit()
            return {"message": "Datos ingresados correctamente"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# 2. Endpoint de reentramiento del modelo
@app.post("/retrain")
async def retrain_model():
    try:
        data = pd.read_sql_query("SELECT * FROM advertising", conn)
        X = data[["TV", "radio", "newspaper"]]
        y = data["sales"]
        model.fit(X, y)
        with open("../model/advertising_model.pkl", "wb") as file:
                pickle.dump(model, file)
        return {'message': 'Modelo reentrenado correctamente.'}
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))

# Carga de FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
