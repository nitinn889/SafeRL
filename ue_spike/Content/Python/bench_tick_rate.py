"""Second, more realistic benchmark: runs reset()/step() from inside a
per-engine-tick callback, so the measurement is bound by the engine's
actual running frame loop rather than a single-shot commandlet with no
loop driving it at all (see ue_bridge.py's run_benchmark() for that
simpler, much-faster-and-less-meaningful number).

Must be launched as a normal (non-commandlet) editor process that stays
alive after the script runs, e.g.:
    UnrealEditor-Cmd Project.uproject -ExecutePythonScript=bench_tick_rate.py -nullrhi -unattended -nosplash -nopause -log
"""
import json
import time

import unreal

from ue_bridge import UEBridge

NUM_STEPS = 500
RESULTS_PATH = "/home/nitin-nandakumar/Downloads/SafeRL/ue_spike/ue_spike_tick_results.json"

_state = {"bridge": None, "count": 0, "t0": None, "handle": None}


def _on_tick(delta_seconds):
    try:
        if _state["bridge"] is None:
            _state["bridge"] = UEBridge()
            _state["bridge"].reset()
            _state["t0"] = time.perf_counter()

        _state["bridge"].step(_state["count"] % 4)
        _state["count"] += 1

        if _state["count"] % 50 == 0 or _state["count"] == 1:
            unreal.log(f"[saferl-spike] tick count={_state['count']}")

        if _state["count"] >= NUM_STEPS:
            elapsed = time.perf_counter() - _state["t0"]
            result = {
                "num_steps": _state["count"],
                "elapsed_seconds": elapsed,
                "steps_per_second": _state["count"] / elapsed if elapsed > 0 else float("inf"),
                "note": "measured from inside the engine's real per-tick callback, "
                        "one env step per engine frame -- bound by actual tick rate, "
                        "not raw call overhead",
            }
            unreal.log(f"[saferl-spike] DONE {result}")
            with open(RESULTS_PATH, "w") as f:
                json.dump(result, f, indent=2)
            unreal.log(f"[saferl-spike] wrote {RESULTS_PATH}")
            unreal.unregister_slate_post_tick_callback(_state["handle"])
            unreal.SystemLibrary.quit_editor()
    except Exception as e:
        unreal.log_error(f"[saferl-spike] EXCEPTION in tick callback: {e!r}")
        import traceback
        unreal.log_error(traceback.format_exc())
        raise


_state["handle"] = unreal.register_slate_post_tick_callback(_on_tick)
unreal.log("[saferl-spike] tick-driven benchmark armed, waiting for engine ticks...")
