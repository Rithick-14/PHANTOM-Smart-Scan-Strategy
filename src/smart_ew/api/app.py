"""
FastAPI REST API Microservice for Smart Scan EW Scheduler.
"""

from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from smart_ew.environment.schema import SimConfig, EmitterConfig
from smart_ew.environment.simulator import RFEnvironment
from smart_ew.emitters.continuous import ContinuousEmitter
from smart_ew.receiver.schema import ReceiverConfig
from smart_ew.receiver.model import ESReceiver
from smart_ew.scheduling.change_adaptive import ChangeAdaptiveScheduler
from smart_ew.evaluation.runner import SimulationRunner


app = FastAPI(
    title="Smart Scan EW Scheduler API",
    description="REST API interface for Electronic Support receiver scheduling and RF spectrum simulation.",
    version="1.0.0",
)

# Global in-memory state
state = {
    "env": RFEnvironment(SimConfig(num_bands=10, total_time_steps=100)),
    "rx": ESReceiver(ReceiverConfig()),
    "scheduler": ChangeAdaptiveScheduler(num_bands=10),
    "runner": SimulationRunner(),
    "step_count": 0,
}


class StepResponse(BaseModel):
    step: int
    scanned_band: int
    detected: bool
    measured_power_dbm: float
    estimated_snr_db: float
    tuning_cost_paid: float


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Smart Scan Strategy for Electronic Warfare API",
        "version": "1.0.0",
    }


@app.post("/simulation/reset")
def reset_simulation(num_bands: int = 10, total_steps: int = 100):
    state["env"] = RFEnvironment(SimConfig(num_bands=num_bands, total_time_steps=total_steps))
    state["env"].add_emitter(ContinuousEmitter(EmitterConfig("C1", "continuous", [1], -50.0)))
    state["rx"] = ESReceiver(ReceiverConfig())
    state["scheduler"] = ChangeAdaptiveScheduler(num_bands=num_bands)
    state["step_count"] = 0
    return {"message": "Simulation reset successfully", "num_bands": num_bands}


@app.post("/simulation/step", response_model=StepResponse)
def execute_step():
    t = state["step_count"]
    frame = state["env"].step()
    action_band = state["scheduler"].select_next_band(t)
    obs = state["rx"].scan(action_band, t, frame)
    state["scheduler"].update(obs)
    state["step_count"] += 1

    return StepResponse(
        step=t,
        scanned_band=obs.band_id,
        detected=obs.detected,
        measured_power_dbm=obs.measured_power_dbm,
        estimated_snr_db=obs.estimated_snr_db,
        tuning_cost_paid=obs.tuning_cost_paid,
    )
