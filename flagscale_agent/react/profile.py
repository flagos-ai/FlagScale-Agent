"""WorkerProfile — parameterizes WorkerAgent behavior by persona.

A WorkerProfile is NOT a separate Agent class. It's a configuration that tells
a single WorkerAgent class which skills to load, which Checklist to use, and
which scene constraints to activate.

Adding a new scenario = registering a new WorkerProfile. No new class needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorkerProfile:
    """Defines a worker persona — not a separate class."""

    name: str
    description: str
    skills: list[str]  # Skills to preload
    scene_constraints: list[str] = field(default_factory=list)
    # checklist is constructed at runtime from skills + scene_constraints
    # so we don't store it here — WorkerAgent builds it during init


# ── Registered profiles ────────────────────────────────────────────────────

PROFILES: dict[str, WorkerProfile] = {
    "general": WorkerProfile(
        name="general",
        description="General-purpose tasks: shell operations, file inspection, simple Q&A, cleanup",
        skills=[],
        scene_constraints=[],
    ),
    "model-migration": WorkerProfile(
        name="model-migration",
        description="Migrate models from source frameworks to FlagScale/Megatron-Core",
        skills=["train-model-porter", "train-config", "train-data-prep"],
        scene_constraints=["is_migration"],
    ),
    "training-reproduce": WorkerProfile(
        name="training-reproduce",
        description="Reproduce training results from papers/reference implementations",
        skills=["train-reproduce", "train-config", "train-run"],
        scene_constraints=["is_training"],
    ),
    "train-env-setup": WorkerProfile(
        name="train-env-setup",
        description="Detect hardware, create environments, install dependencies",
        skills=["train-env-setup", "workspace-layout"],
        scene_constraints=[],
    ),
    "inference-deploy": WorkerProfile(
        name="inference-deploy",
        description="Deploy inference services with vllm/sglang (coming soon)",
        skills=["train-config"],  # future: inference-specific skills
        scene_constraints=["is_inference"],
    ),
}
