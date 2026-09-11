# evaluation package — lazy imports
__all__ = ["BenchmarkSuite", "SafeRLMetrics"]


def __getattr__(name):
    if name == "BenchmarkSuite":
        from .benchmark import BenchmarkSuite
        return BenchmarkSuite
    if name == "SafeRLMetrics":
        from .metrics import SafeRLMetrics
        return SafeRLMetrics
    raise AttributeError(f"module 'evaluation' has no attribute {name!r}")
