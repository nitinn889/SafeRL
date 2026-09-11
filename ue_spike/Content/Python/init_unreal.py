"""Auto-executed by UE's PythonScriptPlugin on editor/engine startup
(any file literally named init_unreal.py under Content/Python is run
automatically -- no ini config needed). Used here only to confirm the
Python environment is alive before running the actual spike script.
"""
import unreal

unreal.log("[saferl-spike] init_unreal.py loaded -- Python bridge is alive.")
