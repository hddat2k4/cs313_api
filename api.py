from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import psycopg2
import pandas as pd
from sqlalchemy import text


# 1. Kết nối PostgreSQL
url = "postgresql://mooc_cs313_user:8sSbCiaKNmRdYyQRaNpwiyefbDbhmRtk@dpg-d0o7mkmmcj7s73e7nmk0-a.singapore-postgres.render.com/mooc_cs313"
engine = create_engine(url)

app = FastAPI()

@app.get("/users/{user_id}")
def get_user(user_id: str):
    query = text('SELECT * FROM "user" WHERE id = :uid')
    with engine.connect() as conn:
        result = conn.execute(query, {"uid": user_id}).fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    return dict(result._mapping)