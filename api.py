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

@app.get("/field/user-counts")
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


@app.get("/course/user-counts")
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

    # return {"table_counts": row_counts}
    return Response(
        content=json.dumps({"table_counts": row_counts}, ensure_ascii=False),
        media_type="application/json; charset=utf-8",
        status_code=status.HTTP_200_OK
    )
    
    
# -------- KGAT Model Endpoints -------
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf_v1
import os

# Tắt eager execution do dùng TF1.x
tf_v1.disable_eager_execution()

# Import KGAT
from model.KGAT import KGAT

CHECKPOINT_DIR = "./kgat"

N_USERS_REAL = 200000
N_ITEMS_REAL = 150000
N_ENTITIES_REAL = 169246
N_RELATIONS_REAL = 10

# Dummy adjacency matrix
dummy_A_in = sp.csr_matrix((N_USERS_REAL + N_ENTITIES_REAL, N_USERS_REAL + N_ENTITIES_REAL), dtype=np.float32)

data_config = {
    'n_users': N_USERS_REAL,
    'n_items': N_ITEMS_REAL,
    'n_entities': N_ENTITIES_REAL,
    'n_relations': N_RELATIONS_REAL,
    'A_in': dummy_A_in,
    'all_h_list': list(range(100)),
    'all_r_list': list(range(100)),
    'all_t_list': list(range(100)),
    'all_v_list': list(range(100))
}

def parse_args_dummy():
    class Args:
        pass
    args = Args()
    args.model_type = 'kgat'
    args.dataset = 'mooccubex-300k-line'
    args.pretrain = 0
    args.adj_type = 'si'
    args.alg_type = 'bi'
    args.embed_size = 32
    args.layer_size = '[64,32,16]'
    args.lr = 0.0001
    args.regs = '[1e-4,1e-4,1e-2]'
    args.node_dropout = '[0.2]'
    args.mess_dropout = '[0.2,0.2,0.2]'
    args.gpu_id = 0
    args.verbose = 50
    args.use_kge = True
    args.use_att = True
    args.adj_uni_type = False
    args.batch_size = 1024
    args.kge_size = 64
    args.batch_size_kg = 2048
    return args

# Request body
class PredictRequest(BaseModel):
    user_id: int
    item_ids: List[int]

@app.post("/predict")
def predict_score(request: PredictRequest) -> Dict[int, float]:
    user_id = request.user_id
    item_ids = request.item_ids

    # Validate
    if user_id >= N_USERS_REAL:
        raise HTTPException(status_code=400, detail="user_id vượt quá giới hạn.")
    if any(item_id >= N_ITEMS_REAL for item_id in item_ids):
        raise HTTPException(status_code=400, detail="Một số item_id vượt quá giới hạn.")

    tf_v1.reset_default_graph()
    tf_v1.set_random_seed(2019)
    np.random.seed(2019)

    args = parse_args_dummy()

    try:
        model = KGAT(data_config=data_config, pretrain_data=None, args=args)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi khởi tạo model: {str(e)}")

    config_proto = tf_v1.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf_v1.Session(config=config_proto)
    saver = tf_v1.train.Saver()

    try:
        ckpt_state = tf_v1.train.get_checkpoint_state(CHECKPOINT_DIR)
        if not ckpt_state or not ckpt_state.model_checkpoint_path:
            raise Exception("Không tìm thấy checkpoint.")
        saver.restore(sess, ckpt_state.model_checkpoint_path)
    except Exception as e:
        sess.close()
        raise HTTPException(status_code=500, detail=f"Lỗi khi khôi phục model: {str(e)}")

    node_dropout_feed = [0.] * len(eval(args.layer_size))
    mess_dropout_feed = [0.] * len(eval(args.layer_size))

    feed_dict = {
        model.users: [user_id],
        model.node_dropout: node_dropout_feed,
        model.mess_dropout: mess_dropout_feed
    }

    try:
        all_entity_scores = sess.run(model.batch_test_scores, feed_dict=feed_dict)[0]
        predictions = {}
        for item_id in item_ids:
            if item_id < model.n_items:
                predictions[item_id] = float(all_entity_scores[item_id])
        sess.close()
        return predictions
    except Exception as e:
        sess.close()
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự đoán: {str(e)}")
    
# Ví dụ gọi API dự đoán:
# curl -X POST "http://localhost:8000/predict" \
#      -H "Content-Type: application/json" \
#      -d '{"user_id": 0, "item_ids": [10, 25, 30]}'\