from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from myagent.rag import RouterAgent
from myagent.guardrails import run_input_guardrails, run_output_guardrails
import os
from dotenv import load_dotenv

load_dotenv()
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

app = FastAPI(title="Math Routing Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = RouterAgent()

class AskReq(BaseModel):
    query: str
    user_id: str = "anonymous"

class AskResp(BaseModel):
    request_id: str
    steps: list
    final_answer: str
    sources: list
    confidence: float

class FeedbackReq(BaseModel):
    request_id: str
    user_id: str = "anonymous"
    rating: int = 0
    corrections: str = ""

@app.post("/ask", response_model=AskResp)
def ask(req: AskReq):
    try:
        run_input_guardrails(req.query)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = agent.handle_query(req.query, requester=req.user_id)

    try:
        run_output_guardrails(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Output failed guardrails: " + str(e))

    return AskResp(
        request_id=result.get("request_id", "local-"+req.user_id),
        steps=result.get("steps", []),
        final_answer=result.get("final_answer", ""),
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0)
    )

@app.post("/feedback")
def feedback(payload: FeedbackReq):
    # Simple: store feedback locally -> append to file feedback.log
    import json
    entry = {
        "request_id": payload.request_id,
        "user_id": payload.user_id,
        "rating": payload.rating,
        "corrections": payload.corrections
    }
    with open("feedback.log","a") as f:
        f.write(json.dumps(entry)+"\n")
    # Signal: DSPy or manual process can pick up this file to update KB/rerank later
    return {"status":"saved"}

# Simple health endpoint (added automatically)
@app.get("/health")
def health():
    return {"status":"ok"}
