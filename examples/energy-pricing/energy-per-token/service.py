"""Serves each region's electricity price as an Orla cost source.

One endpoint per grid hub, each answering the contract Orla polls for:

    GET /energy-per-token/caiso-np15
    {"joules_per_token": 0.021}

Energy per token estimates come from TokenPowerBench

Run it with `just energy` (uvicorn on :8083).

Environment: DATA (path to the hub CSV), JOULES_PER_PROMPT_TOKEN,
JOULES_PER_COMPLETION_TOKEN.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

DATA = Path(os.environ.get("DATA", "../data/TokenPowerBench_averaged_metric.csv"))


# Energy per token for each model, assuming engine vLLM, batch size 512, output length 2000
# What about prefill vs decode differences??? does TokenBench only show decode?
MODELS = {
    "Llama-3.2-1B": 0.0434169026052989,
    "Llama-3.2-3B": 0.0761157123324893,
    "Llama-3.1-8B": 0.108132992100189,
}


class EnergyEfficiency(BaseModel):
    """The energy efficiency contract Orla polls for."""

    joules_per_token: float


app = FastAPI(title="orla-energy-per-token-example")


@app.get("/energy-per-token/{model}", response_model=EnergyEfficiency)
def energy_per_token(model: str) -> EnergyEfficiency:
    if model not in MODELS:
        raise HTTPException(status_code=404, detail=f"unknown model {model}")
    return EnergyEfficiency(joules_per_token=MODELS[model])


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
