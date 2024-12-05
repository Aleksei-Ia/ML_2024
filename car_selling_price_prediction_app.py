from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import io
from fastapi.responses import StreamingResponse
from fastapi import File, UploadFile

app = FastAPI()

with open('model.pkl', 'rb') as f:
    gsZ = pickle.load(f)

class Item(BaseModel):
    name: str
    year: int
    selling_price: int = None
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

@app.post('/predict_item')
def predict_item(item: Item) -> dict:
    data = pd.DataFrame([item.dict()])
    processed = preprocess_data(data)
    prediction = gsZ.predict(processed)
    return {'predictions': round(abs(prediction[0]), 2)}

@app.post("/predict_csv", summary="Upload CSV File for Prediction", description="""
Загрузите CSV файл, содержащий данные автомобилей, для получения предсказаний о цене.

Структура файла должна содержать следующие столбцы:

- name (str): Название автомобиля, например Lamborghini Gallardo
- year (int): Год выпуска
- selling_price (int, optional): Цена продажи (столбец может отсутствовать)
- km_driven (int): Пробег автомобиля
- fuel (str): Тип топлива (Petrol, Diesel, LPG, CNG)
- seller_type (str): Тип продавца, например Dealer
- transmission (str): Тип трансмиссии, например Manual
- owner (str): Количество предыдущих владельцев, например First Owner
- mileage (str): Расход топлива (kmpl, km/kg)
- engine (str): Объем двигателя в CC
- max_power (str): Максимальная мощность в bhp
- torque (str): Крутящий момент в NM или kgm при оборотах, например 190Nm 2000rpm
- seats (float): Количество мест
""")
def predict_csv(file: UploadFile = File(...)) -> StreamingResponse:
    try:
        df = pd.read_csv(file.file)
        if 'selling_price' not in df.columns:
            df['selling_price'] = np.nan
        processed = preprocess_data(df.copy())
        predictions = gsZ.predict(processed)
        df['predictions'] = [round(abs(x), 2) for x in predictions]
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type='text/csv')
        response.headers['Content-Disposition'] = 'attachment; filename=result.csv'
        return response
    except Exception as e:
        return {'error': str(e)}

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    def convert_mileage(mileage, fuel):
        if 'km/kg' in str(mileage):
            mileage_value = float(mileage.replace(' km/kg', ''))
            if fuel == 'Petrol':
                return round(mileage_value * 0.75, 2)
            elif fuel == 'Diesel':
                return round(mileage_value * 0.85, 2)
            elif fuel == 'LPG':
                return round(mileage_value * 0.55, 2)
            elif fuel == 'CNG':
                return round(mileage_value * 0.72, 2)
        return mileage
    df['mileage'] = df.apply(lambda row: convert_mileage(row['mileage'], row['fuel']), axis=1)
    df['mileage'] = df['mileage'].apply(lambda x: float(x.replace(' kmpl', '')) if isinstance(x, str) else x)
    df['engine'] = df['engine'].apply(lambda x: float(x.replace(' CC', '')) if isinstance(x, str) else x)
    df['max_power'] = df['max_power'].replace(' bhp', np.nan).apply(lambda x: float(x.replace(' bhp', '')) if isinstance(x, str) else x)
    df['torque'] = df['torque'].apply(lambda x: x.lower().replace('@', ' ').replace('rpm', ' ').replace('at', ' ').replace('nm', ' ').replace('(', ' ').replace(')', ' ').replace('+/-500', '').replace('/', '').replace('~', '-').replace(',', '').replace('kgm', ' kgm') if isinstance(x, str) else x)
    df['torque'] = df['torque'].apply(lambda x: ' '.join([str(round(float(value) * 9.80665, 2)) if i == 0 and 'kgm' in x else value for i, value in enumerate(x.split())]).replace('kgm', '') if isinstance(x, str) and 'kgm' in x else x)
    df['torque'] = [item.split()[0] + ' ' + (lambda b: str((int(b.split('-')[0]) + int(b.split('-')[1])) // 2) if '-' in b else b)(item.split()[1]) if isinstance(item, str) and len(item.split()) == 2 else item for item in df['torque']]
    df['torque'] = [float(round((float(item.split()[0]) * ((int(c.split('-')[0]) + int(c.split('-')[1])) // 2 if '-' in (c := item.split()[1]) else int(c)) * 2 * np.pi) / 60, 2)) if isinstance(item, str) and len(item.split()) == 2 else np.nan for item in df['torque']]
    for col in df:
        if df[col].isna().sum() > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
    df['engine'] = df['engine'].astype(int)
    df['seats'] = df['seats'].astype(int)
    df['model'] = df['name'].apply(lambda x: x.split()[1] if len(x.split()) > 1 else 'Unknown')
    df['name'] = df['name'].apply(lambda x: ' '.join(x.split()[:1]))
    df = pd.get_dummies(df, columns=df.select_dtypes(include='object').columns, drop_first=True)
    model_columns = gsZ.feature_names_in_
    missing_cols = set(model_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    df = df.reindex(columns=model_columns, fill_value=0)
    return df