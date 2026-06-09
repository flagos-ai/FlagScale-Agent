"""Explicit state machine for Agent lifecycle.

Replaces implicit phase strings with a proper FSM.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal


class AgentState(Enum):
    """Agent lifecycle states."""

    IDLE = auto()           # Initial state, waiting for user input
    PLANNING = auto()       # Analyzing task, building plan
    EXECUTING = auto()      # Running tools, making progress
    REVIEWING = auto()      # Strategic review, checking progress
    WAITING_USER = auto()   # Blocked on user input/confirmation
    COMPACTING = auto()     # Context compression in progress
    COMPLETED = auto()      # Task successfully completed
    FAILED = auto()         # Unrecoverable error
    INTERRUPTED = auto()    # User interrupted (Ctrl+C)


@dataclass
class StateTransition:
    """A state transition with metadata."""

    from_state: AgentState
    to_state: AgentState
    reason: str = ""
    metadata: dict | None = None


class StateMachine:
    """FSM for agent lifecycle with transition validation."""

    # Valid transitions: from_state -> set of allowed to_states
    _TRANSITIONS = {
        AgentState.IDLE: {
            AgentState.PLANNING,
            AgentState.EXECUTING,
            AgentState.INTERRUPTED,
        },
        AgentState.PLANNING: {
            AgentState.EXECUTING,
            AgentState.WAITING_USER,
            AgentState.FAILED,
            AgentState.INTERRUPTED,
        },
        AgentState.EXECUTING: {
            AgentState.EXECUTING,      # Continue execution
            AgentState.REVIEWING,
            AgentState.WAITING_USER,
            AgentState.COMPACTING,
            AgentState.COMPLETED,
            AgentState.FAILED,
            AgentState.INTERRUPTED,
        },
        AgentState.REVIEWING: {
            AgentState.PLANNING,       # Re-plan after review
            AgentState.EXECUTING,      # Continue execution
            AgentState.COMPLETED,
            AgentState.FAILED,
            AgentState.INTERRUPTED,
        },
        AgentState.WAITING_USER: {
            AgentState.PLANNING,
            AgentState.EXECUTING,
            AgentState.INTERRUPTED,
        },
        AgentState.COMPACTING: {
            AgentState.EXECUTING,      # Resume after compaction
            AgentState.FAILED,
            AgentState.INTERRUPTED,
        },
        AgentState.COMPLETED: set(),   # Terminal state
        AgentState.FAILED: set(),      # Terminal state
        AgentState.INTERRUPTED: set(), # Terminal state
    }

    def __init__(self, initial_state: AgentState = AgentState.IDLE):
        self.current_state = initial_state
        self.history: list[StateTransition] = []

    def can_transition(self, to_state: AgentState) -> bool:
        """Check if transition is valid."""
        return to_state in self._TRANSITIONS.get(self.current_state, set())

    def transition(self, to_state: AgentState, reason: str = "", metadata: dict | None = None) -> bool:
        """Attempt state transition. Returns True if successful."""
        if not self.can_transition(to_state):
            return False

        transition = StateTransition(
            from_state=self.current_state,
            to_state=to_state,
            reason=reason,
            metadata=metadata,
        )
        self.history.append(transition)
        self.current_state = to_state
        return True

    def force_transition(self, to_state: AgentState, reason: str = "", metadata: dict | None = None):
        """Force transition without validation (use sparingly)."""
        transition = StateTransition(
            from_state=self.current_state,
            to_state=to_state,
            reason=f"FORCED: {reason}",
            metadata=metadata,
        )
        self.history.append(transition)
        self.current_state = to_state

    def is_terminal(self) -> bool:
        """Check if current state is terminal."""
        return self.current_state in {
            AgentState.COMPLETED,
            AgentState.FAILED,
            AgentState.INTERRUPTED,
        }

    def get_phase_name(self) -> str:
        """Get phase name for backward compatibility with old code."""
        # Map states to old phase strings
        _STATE_TO_PHASE = {
            AgentState.IDLE: "idle",
            AgentState.PLANNING: "planning",
            AgentState.EXECUTING: "executing",
            AgentState.REVIEWING: "reviewing",
            AgentState.WAITING_USER: "waiting_user",
            AgentState.COMPACTING: "compacting",
            AgentState.COMPLETED: "completed",
            AgentState.FAILED: "failed",
            AgentState.INTERRUPTED: "interrupted",
        }
        return _STATE_TO_PHASE.get(self.current_state, "unknown")

    @classmethod
    def from_phase_name(cls, phase_name: str) -> StateMachine:
        """Create FSM from old phase string (for migration)."""
        _PHASE_TO_STATE = {
            "idle": AgentState.IDLE,
            "planning": AgentState.PLANNING,
            "executing": AgentState.EXECUTING,
            "reviewing": AgentState.REVIEWING,
            "waiting_user": AgentState.WAITING_USER,
            "compacting": AgentState.COMPACTING,
            "completed": AgentState.COMPLETED,
            "failed": AgentState.FAILED,
            "interrupted": AgentState.INTERRUPTED,
        }
        state = _PHASE_TO_STATE.get(phase_name, AgentState.IDLE)
        return cls(initial_state=state)
