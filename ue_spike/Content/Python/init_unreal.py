"""Auto-executed by UE's PythonScriptPlugin on editor/engine startup
(any file literally named init_unreal.py under Content/Python is run
automatically -- no ini config needed).

When SAFERL_RUN_PIE=1 is set, this arms the phase 3 PIE session
(pie_session.arm()). Startup scripts run early, so arm() only registers a
tick callback; the scene build and PIE request happen on later ticks.

This is deliberately the entrypoint instead of -ExecutePythonScript: that
flag makes the editor quit as soon as the script returns, which is what
killed the phase 2 tick-callback attempt.
"""
import os

import unreal

unreal.log("[saferl-spike] init_unreal.py loaded -- Python bridge is alive.")

if os.environ.get("SAFERL_RUN_PIE") == "1":
    import pie_session
    pie_session.arm()
