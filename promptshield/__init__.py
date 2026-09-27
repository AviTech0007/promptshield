"""PromptShield: a two-checkpoint firewall for AI agents.

    from promptshield import Shield, ActionHeld
"""
from .shield import ActionHeld, Shield
from .types import GateDecision, ScanResult

__all__ = ["Shield", "ActionHeld", "ScanResult", "GateDecision"]
__version__ = "0.9.0"
