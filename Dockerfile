FROM python:3.11-slim

WORKDIR /code

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs build-essential \
    ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# --- Python dependencies ---
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /code/requirements.txt

# --- Copy project files ---
COPY . /code/

# --- Expose port for FastAPI ---
EXPOSE 8000

# --- Start FastAPI app ---
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
