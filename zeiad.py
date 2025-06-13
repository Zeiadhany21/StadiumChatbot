from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import psycopg2
import os
from dotenv import load_dotenv

# Load DB credentials
load_dotenv()
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "mydatabase")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# --------------------- Database Access ---------------------
def get_facilities():
    conn = psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cur = conn.cursor()
    cur.execute("SELECT id FROM \"Facilities\" ORDER BY id")
    facilities = [row[0] for row in cur.fetchall()]
    conn.close()
    return facilities

def get_sports():
    conn = psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cur = conn.cursor()
    cur.execute("SELECT id FROM \"Sports\" ORDER BY id")
    sports = [row[0] for row in cur.fetchall()]
    conn.close()
    return sports

# --------------------- ID Mappings ---------------------
FACILITY_IDS = get_facilities()
SPORT_IDS = get_sports()

facility2idx = {fac: idx for idx, fac in enumerate(FACILITY_IDS)}
idx2facility = {idx: fac for fac, idx in facility2idx.items()}
sport2idx = {sport: idx for idx, sport in enumerate(SPORT_IDS)}

# --------------------- Model ---------------------
class FacilityRecommenderNet(nn.Module):
    def __init__(self, n_sports, n_facilities):
        super().__init__()
        self.sport_embed = nn.Embedding(n_sports, 8)
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, n_facilities)

    def forward(self, x):
        sport = x[:, 0]
        sport_emb = self.sport_embed(sport)
        x = F.relu(self.fc1(sport_emb))
        return self.fc2(x)

# ---------------------  Training ---------------------
def train_model():
    model = FacilityRecommenderNet(n_sports=len(SPORT_IDS), n_facilities=len(FACILITY_IDS))
    model.eval()
    return model

model = train_model()

# --------------------- FastAPI ---------------------
app = FastAPI()

class RecommendRequest(BaseModel):
    sport_id: int

class RecommendResponse(BaseModel):
    facility_ids: list[int]
    scores: list[float]

@app.post("/recommend", response_model=RecommendResponse)
def recommend_facilities(req: RecommendRequest):
    sport_idx = sport2idx.get(req.sport_id, 0)

    x = torch.tensor([[sport_idx]], dtype=torch.long)
    with torch.no_grad():
        scores = model(x)
        top_scores, top_indices = torch.topk(scores, 3, dim=1)

    facility_ids = [idx2facility[i.item()] for i in top_indices[0]]
    score_vals = [round(s.item(), 4) for s in top_scores[0]]

    return RecommendResponse(facility_ids=facility_ids, scores=score_vals)

# To run: uvicorn facility_recommender_api:app --reload
# POST to http://127.0.0.1:8000/recommend with JSON:
# {"sport_id": 1}
