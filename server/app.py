"""
FastAPI HTTP server — wraps WitnessStandEnv for OpenEnv HTTP interface.
Endpoints: /reset /step /score /transcript
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from environment import WitnessStandEnv
from models import WitnessAction

app = FastAPI(title="The Witness Stand", version="1.0.0")
_env = WitnessStandEnv()


class ResetRequest(BaseModel):
    task_name: str = "basic"


class StepRequest(BaseModel):
    response_text: str
    flagged_distortion: bool = False
    accepted_update: bool = False


@app.post("/reset")
def reset(req: ResetRequest):
    try:
        obs = _env.reset(req.task_name)
        return {"observation": obs, "done": False}
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown task: {req.task_name}")


@app.post("/step")
def step(req: StepRequest):
    action = WitnessAction(
        response_text=req.response_text,
        flagged_distortion=req.flagged_distortion,
        accepted_update=req.accepted_update,
    )
    obs, reward, done, info = _env.step(action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}


@app.get("/score")
def score():
    if _env.episode_log is None:
        raise HTTPException(status_code=400, detail="No episode in progress.")
    return {
        "per_turn_scores": _env.episode_log.per_turn_scores,
        "episode_score": _env.episode_log.episode_score,
        "final_score": _env.episode_log.final_score,
    }


@app.get("/transcript")
def transcript():
    if _env.transcript is None:
        raise HTTPException(status_code=400, detail="No episode in progress.")
    return {"turns": [t.__dict__ for t in _env.transcript.get_all()]}


@app.get("/")
def health():
    return {"status": "ok", "environment": "witness_stand"}
