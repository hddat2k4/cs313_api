from fastapi import FastAPI, HTTPException, Response, status
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
import psycopg2
from sqlalchemy import text
import json

# 1. Kết nối PostgreSQL
url = "postgresql://mooc_cs313_user:8sSbCiaKNmRdYyQRaNpwiyefbDbhmRtk@dpg-d0o7mkmmcj7s73e7nmk0-a.singapore-postgres.render.com/mooc_cs313"
engine = create_engine(url)

app = FastAPI()

@app.get("/user")
def get_user(id: str = None, name: str = None): 
    with engine.connect() as conn:
        if id:
            query = text('SELECT id, name, gender, school, course_order FROM "user" WHERE id = :id')
            result = conn.execute(query, {"id": id}).fetchall()
        elif name:
            query = text('SELECT id, name, gender, school, course_order FROM "user" WHERE name ILIKE :name')
            result = conn.execute(query, {"name": f"%{name}%"}).fetchall()
        else:
            query = text('SELECT id, name, gender, school, course_order FROM "user"')
            result = conn.execute(query).fetchall()
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    # return [dict(row._mapping) for row in result]
    return Response(
        content=json.dumps({"data": [dict(row._mapping) for row in result]}, ensure_ascii=False),
        media_type="application/json; charset=utf-8",
        status_code=status.HTTP_200_OK
    )

@app.get("/course")
def get_courses(id: str = None, name: str = None):
    with engine.connect() as conn:
        if id:
            query = text('SELECT course_id, name, start_date, end_date, field FROM "course" WHERE id = :id')
            result = conn.execute(query, {"id": id}).fetchall()
        elif name:
            query = text('SELECT course_id, name, start_date, end_date, field FROM "course" WHERE name ILIKE :name')
            result = conn.execute(query, {"name": f"%{name}%"}).fetchall()
        else:
            query = text('SELECT course_id, name, start_date, end_date, field FROM "course"')
            result = conn.execute(query).fetchall()

    if not result:
        raise HTTPException(status_code=404, detail="Course not found")
    return Response(
        content=json.dumps({"data": [dict(row._mapping) for row in result]}, ensure_ascii=False),
        media_type="application/json; charset=utf-8",
        status_code=status.HTTP_200_OK
    )


@app.get("/teacher")
def get_teacher(id: str = None, name: str = None):
    with engine.connect() as conn:
        if id:
            query = text('SELECT id, name_en, job_title FROM "teacher" WHERE id = :id')
            result = conn.execute(query, {"id": id}).fetchall()
        elif name:
            query = text('SELECT id, name_en, job_title FROM "teacher" WHERE name_en ILIKE :name')
            result = conn.execute(query, {"name": f"%{name}%"}).fetchall()
        else:
            query = text('SELECT id, name_en, job_title FROM "teacher"')
            result = conn.execute(query).fetchall()

    if not result:
        raise HTTPException(status_code=404, detail="No teachers found")
    return Response(
        content=json.dumps({"data": [dict(row._mapping) for row in result]}, ensure_ascii=False),
        media_type="application/json; charset=utf-8",
        status_code=status.HTTP_200_OK
    )


@app.get("/field")
def get_field(name: str = None):
    with engine.connect() as conn:
        if name:
            query = text('SELECT * FROM "field" WHERE field ILIKE :name')
            result = conn.execute(query, {"name": f"%{name}%"}).fetchall()
        else:
            query = text('SELECT * FROM "field"')
            result = conn.execute(query).fetchall()

    if not result:
        raise HTTPException(status_code=404, detail="Field not found")
    return Response(
        content=json.dumps({"data": [dict(row._mapping) for row in result]}, ensure_ascii=False),
        media_type="application/json; charset=utf-8",
        status_code=status.HTTP_200_OK
    )

@app.get("/users-per-field")
def users_per_field():
    query = text("""
        SELECT field_value, COUNT(DISTINCT user_id) AS num_users
        FROM (
            SELECT user_id, jsonb_array_elements_text(field::jsonb) AS field_value
            FROM user_course
        ) exploded
        GROUP BY field_value
        ORDER BY num_users DESC
    """)
    with engine.connect() as conn:
        result = conn.execute(query).mappings()
        return Response(
        content=json.dumps({"data": [{"field": row["field_value"], "num_users": row["num_users"]} for row in result]}, ensure_ascii=False),
        media_type="application/json; charset=utf-8",
        status_code=status.HTTP_200_OK
    )


@app.get("/users-per-course")
def users_per_course():
    with engine.connect() as conn:
        query = text("""
            SELECT course_id, COUNT(*) AS num_users
            FROM "user_course"
            GROUP BY course_id
            ORDER BY num_users DESC
        """)
        result = conn.execute(query).mappings()
        return Response(
        content=json.dumps({"data": [{"course_id": row["course_id"], "user_count": row["num_users"]} for row in result]}, ensure_ascii=False),
        media_type="application/json; charset=utf-8",
        status_code=status.HTTP_200_OK
        )


@app.get("/summary")
def get_summary():
    row_counts = {}
    with engine.connect() as conn:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        for table in tables:
            try:
                result = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"'))
                count = result.scalar()
                row_counts[table] = count
            except Exception as e:
                row_counts[table] = f"Error: {str(e)}"
    
    if not row_counts:
        raise HTTPException(status_code=404, detail="No tables found or failed to count rows")

    return {"table_counts": row_counts}
    return Response(
        content=json.dumps({"table_counts": row_counts}, ensure_ascii=False),
        media_type="application/json; charset=utf-8",
        status_code=status.HTTP_200_OK
    )