# CS313 Course Recommendation API

This project is a backend system for course recommendation and analytics, built with **FastAPI** and powered by an AI model based on the **Knowledge Graph Attention Network (KGAT)**. It provides RESTful APIs for querying users, courses, teachers, fields, and for generating personalized course recommendations.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [AI Model: KGAT](#ai-model-kgat)
- [Setup & Installation](#setup--installation)
- [API Endpoints](#api-endpoints)
- [Example Usage](#example-usage)
- [License](#license)

---

## Features

- **User, Course, Teacher, Field APIs:** Query and analyze educational data stored in PostgreSQL.
- **Analytics:** Get user counts per field/course, and database summary statistics.
- **AI-powered Recommendation:** Personalized course recommendations using the KGAT model.
- **Batch Prediction:** Predict scores for a user and a list of items.
- **Docker Support:** Easy deployment with Docker.

---

## Project Structure

```
.
├── api.py                # Main FastAPI application
├── model/
│   └── KGAT.py           # KGAT model implementation (TensorFlow 1.x)
├── kgat/                 # KGAT model checkpoints
│   ├── checkpoint
│   ├── weights-489.data-00000-of-00001
│   ├── weights-489.index
│   └── weights-489.meta
├── data/                 # Parquet data files
│   ├── course_df.parquet
│   ├── field_df.parquet
│   ├── full_maped_data.parquet
│   ├── school_df.parquet
│   ├── teacher_df.parquet
│   └── user.parquet
├── push_data.ipynb       # Notebook for pushing data to PostgreSQL
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker build file
├── .gitignore
├── .dockerignore
└── README.md             # This file
```

---

## Technologies Used

- **FastAPI**: High-performance Python web framework for building APIs.
- **SQLAlchemy**: Database ORM for PostgreSQL.
- **TensorFlow 1.x**: Deep learning framework for the KGAT model.
- **NumPy, SciPy**: Scientific computing libraries.
- **gdown**: Downloading model weights from Google Drive.
- **Docker**: Containerization for easy deployment.

---

## AI Model: KGAT

The core recommendation engine is based on the **Knowledge Graph Attention Network (KGAT)**, a state-of-the-art model for recommendation systems that leverages both user-item interactions and knowledge graph information.

- **Paper**: [KGAT: Knowledge Graph Attention Network for Recommendation (KDD 2019)](https://arxiv.org/abs/1905.07854)
- **Key Features**:
  - Learns user and item representations by propagating information over a knowledge graph.
  - Uses attention mechanisms to weigh the importance of different relations.
  - Supports both collaborative filtering and knowledge graph embedding.

The model is implemented in [`model/KGAT.py`](model/KGAT.py) and is loaded with pretrained weights from the [`kgat/`](kgat/) directory.

---

## Setup & Installation

### 1. Clone the repository

```sh
git clone <your-repo-url>
cd cs313_api
```

### 2. Install dependencies

```sh
pip install -r requirements.txt
```

### 3. Prepare KGAT model weights

Ensure the `kgat/` directory contains the TensorFlow checkpoint files:
- `checkpoint`
- `weights-489.data-00000-of-00001`
- `weights-489.index`
- `weights-489.meta`

If not present, download and extract them as described in the code.

### 4. Configure Database

Update the PostgreSQL connection string in [`api.py`](api.py) if needed.

### 5. Run the API

```sh
uvicorn api:app --reload
```

Or with Docker:

```sh
docker build -t cs313_api .
docker run -p 8000:8000 cs313_api
```

---

## API Endpoints

### Data APIs

- `GET /user` — Query users by id or name.
- `GET /course` — Query courses by id or name.
- `GET /teacher` — Query teachers by id or name.
- `GET /field` — Query fields by name.
- `GET /field/user-counts` — Number of users per field.
- `GET /course/user-counts` — Number of users per course.
- `GET /summary` — Row counts for all tables.

### Recommendation APIs

- `POST /predict`
  - **Request body:**  
    ```json
    { "user_id": 0, "item_ids": [10, 25, 30] }
    ```
  - **Response:**  
    ```json
    { "10": 0.123, "25": 0.456, "30": 0.789 }
    ```

- `POST /recommend`
  - **Request body:**  
    ```json
    { "user_id": 5, "top_k": 10 }
    ```
  - **Response:**  
    ```json
    { "12": 0.95, "7": 0.93, ... }
    ```

---

## Example Usage

**Predict scores:**
```sh
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"user_id": 0, "item_ids": [10, 25, 30]}'
```

**Get recommendations:**
```sh
curl -X POST "http://localhost:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{"user_id": 5, "top_k": 10}'
```

---

## License

This project is for academic and research purposes.

---

**Author:**  
Tran Thanh Nhan - 22521007