# agents package — lazy imports
__all__ = ["PPOAgent", "SACAgent", "ShieldedAgent"]


def __getattr__(name):
    if name == "PPOAgent":
        from .ppo_agent import PPOAgent
        return PPOAgent
    if name == "SACAgent":
        from .sac_agent import SACAgent
        return SACAgent
    if name == "ShieldedAgent":
        from .shielded_agent import ShieldedAgent
        return ShieldedAgent
    raise AttributeError(f"module 'agents' has no attribute {name!r}")
