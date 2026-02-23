import os
import json
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import JSONResponse
from typing import List, Optional, Any
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://app:app@localhost:5432/cases")
API_KEY = os.getenv("APP_API_KEY", "dev-key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Case Learning Backend", version="0.1.0")

def require_key(x_api_key: Optional[str]):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

def db():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)

def embed(text: str) -> List[float]:
    """
    Creates an embedding using OpenAI's embeddings endpoint.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set in .env")

    import requests
    r = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "text-embedding-3-small",
            "input": text[:8000],
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]

# ---------- Schemas ----------
class UserFeedback(BaseModel):
    rating: Optional[int] = None
    comment: Optional[str] = None

class AddCaseIn(BaseModel):
    title: str
    domain: str
    problem: str
    constraints: str
    best_answer: str
    pitfalls: str
    tags: List[str]

class CaseIn(BaseModel):
    title: str
    domain: str = "general"
    problem: str
    constraints: str = ""
    best_answer: str
    pitfalls: str = ""
    tags: List[str] = Field(default_factory=list)

class SearchIn(BaseModel):
    query: str
    domain: Optional[str] = None
    top_k: int = 5


class CaseOut(BaseModel):
    id: int
    title: str
    domain: str
    card_text: str
    score: float

class SearchOut(BaseModel):
    retrieval_confidence: float
    cases: List[CaseOut]

class LogIn(BaseModel):
    conversation_id: Optional[str] = None
    query: str
    answer: str
    case_id: int
    user_feedback: Optional[UserFeedback] = None
    
class IdResponse(BaseModel):
    id: int

class CaseOut(BaseModel):
    id: int
    title: str
    domain: str
    card_text: str
    score: float

class SearchOut(BaseModel):
    retrieval_confidence: float
    cases: List[CaseOut]

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/add_case", response_model=IdResponse)
def add_case(payload: AddCaseIn, x_api_key: Optional[str] = Header(default=None)):
    require_key(x_api_key)

    card_text = (
        f"Title: {payload.title}\n"
        f"Domain: {payload.domain}\n"
        f"Problem: {payload.problem}\n"
        f"Constraints: {payload.constraints}\n"
        f"Best answer: {payload.best_answer}\n"
        f"Pitfalls: {payload.pitfalls}\n"
        f"Tags: {', '.join(payload.tags)}"
    )
    vec = embed(card_text)

    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO cases (title, domain, problem, constraints, best_answer, pitfalls, tags, embedding)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
            """,
            (payload.title, payload.domain, payload.problem, payload.constraints,
             payload.best_answer, payload.pitfalls, payload.tags, vec),
        )
        new_id = cur.fetchone()["id"]
    return {"id": new_id}


@app.post("/search_cases", response_model=SearchOut)
def search_cases(payload: SearchIn, x_api_key: Optional[str] = Header(default=None)):
    require_key(x_api_key)

    qvec = embed(payload.query)

    where = "TRUE"
    params: List[Any] = [qvec, qvec, payload.top_k]

    if payload.domain:
        where = "domain = %s"
        params = [payload.domain, qvec, qvec, payload.top_k]

    sql = f"""
      SELECT
        id, title, domain,
        (1 - (embedding <=> (%s)::vector)) AS score,
        problem, constraints, best_answer, pitfalls, tags
      FROM cases
      WHERE {where}
      ORDER BY embedding <=> (%s)::vector
      LIMIT %s
    """

    with db() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    cases: List[CaseOut] = []
    for r in rows:
        card = (
            f"ID: {r['id']}\nTitle: {r['title']}\nDomain: {r['domain']}\n"
            f"Problem: {r['problem']}\nConstraints: {r['constraints']}\n"
            f"Best answer: {r['best_answer']}\nPitfalls: {r['pitfalls']}\n"
            f"Tags: {', '.join(r['tags'] or [])}"
        )
        cases.append(CaseOut(
            id=r["id"],
            title=r["title"],
            domain=r["domain"],
            card_text=card,
            score=float(r["score"])
        ))

    best = max([c.score for c in cases], default=0.0)
    confidence = float(max(0.0, min(1.0, best)))

    return {"retrieval_confidence": confidence, "cases": cases}


@app.post("/log_interaction", response_model=IdResponse)
def log_interaction(payload: LogIn, x_api_key: Optional[str] = Header(default=None)):
    require_key(x_api_key)
    feedback_text = json.dumps(payload.user_feedback.model_dump()) if payload.user_feedback else None
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO interactions
              (conversation_id, user_query, assistant_answer, retrieved_case_ids, resolved, followups_count, user_feedback)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
            """,
            (
                payload.conversation_id,
                payload.user_query,
                payload.assistant_answer,
                payload.retrieved_case_ids,
                payload.resolved,
                payload.followups_count,
                feedback_text,
            ),
        )
        new_id = cur.fetchone()["id"]
    return {"id": new_id}
