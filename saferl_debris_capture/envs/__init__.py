# envs package — lazy imports allow sub-modules to be used independently
# without requiring Isaac Lab or torch to be installed.
__all__ = ["DebrisCaptureEnv", "TumblingDebrisDynamics", "RewardShaper"]


def __getattr__(name):
    if name == "DebrisCaptureEnv":
        from .debris_capture_env import DebrisCaptureEnv
        return DebrisCaptureEnv
    if name == "TumblingDebrisDynamics":
        from .debris_dynamics import TumblingDebrisDynamics
        return TumblingDebrisDynamics
    if name == "RewardShaper":
        from .reward_shaping import RewardShaper
        return RewardShaper
    raise AttributeError(f"module 'envs' has no attribute {name!r}")
