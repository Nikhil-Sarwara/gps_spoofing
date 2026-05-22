"""Helper functions for managing GPS spoofing research components.

Provides the interface layer between the TUI and the underlying system
processes. Handles starting/stopping PX4 SITL, GPS monitor, live inference,
dashboard (Streamlit), and Gazebo simulation. Also exposes ML pipeline
launching and data-file discovery for the status panel.
"""

import os
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from .config import COMPONENTS, PROJECT_ROOT, ML_PIPELINE, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from .process_manager import (
    is_process_running,
    get_pid,
    save_pid,
    remove_pid,
    stop_process as pm_stop_process,
)


def start_component(component_id):
    """Start a component by ID in a new window or as a background process.

    Each component is configured in COMPONENTS with a command, working
    directory, log file, and spawn method. ``terminal`` spawns a new macOS
    Terminal.app window via AppleScript; ``background`` forks a subprocess
    managed by the TUI.

    Args:
        component_id: Key into the COMPONENTS configuration dict (e.g.
            ``"px4_sim"``, ``"gps_monitor"``).

    Returns:
        Tuple of ``(success: bool, message: str)``.
    """
    config = COMPONENTS[component_id]
    pid = get_pid(component_id)

    if pid and is_process_running(pid):
        return False, f"{config['name']} is already running"

    # Ensure the log directory exists before the process writes to it.
    log_file = config["log_file"]
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = config["command"]
    workdir = str(config["workdir"])
    spawn_method = config.get("spawn_method", "terminal")
    open_browser = config.get("open_browser", False)

    if spawn_method == "terminal":
        try:
            # macOS-specific: use AppleScript to open a Terminal window and run
            # the command inside the correct working directory.
            workdir_escaped = workdir.replace("'", "'\\''")
            cmd_escaped = cmd.replace("'", "'\\''")

            script = f'''
            tell application "Terminal"
                activate
                do script "cd '{workdir_escaped}' && {cmd_escaped}"
            end tell
            '''
            subprocess.run(["osascript", "-e", script], check=True)

            # Brief pause to let the Terminal / process initialise before
            # returning control to the TUI.
            time.sleep(2)

            msg = f"Started {config['name']} in new window"
            if open_browser:
                time.sleep(3)
                webbrowser.open("http://localhost:8501")
                msg += " - browser opened"

            return True, msg

        except Exception as e:
            return False, f"Error: {e}"
    else:
        try:
            # Background spawn: fork a managed subprocess whose output is
            # captured in a log file so the TUI does not block.
            log_f = log_file.open("a")

            # HACK: Create a new process group via os.setsid so that killing
            # the process group later also kills any child processes spawned
            # by the shell (e.g. `python -m ...` launched via shell=True).
            # Not supported on Windows.
            env = os.environ.copy()

            process = subprocess.Popen(
                cmd,
                cwd=workdir,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid if os.name != "nt" else None,
                shell=True,
            )

            save_pid(component_id, process.pid)

            if open_browser:
                time.sleep(3)
                webbrowser.open("http://localhost:8501")

            return True, f"Started {config['name']} (PID: {process.pid})"
        except Exception as e:
            return False, f"Error: {e}"


def stop_component(component_id):
    """Stop a component by ID using process-name matching and PID kill.

    First attempts a ``pkill`` match on the component's name plus a set of
    well-known names (``px4``, ``gps_monitor``, ``live_inference``,
    ``streamlit``) to catch stray child processes. Falls back to killing the
    tracked PID via :func:`process_manager.stop_process`.

    Args:
        component_id: Key into the COMPONENTS configuration dict.

    Returns:
        Tuple of ``(success: bool, message: str)``.
    """
    pid = get_pid(component_id)
    config = COMPONENTS[component_id]

    # Broad pkill sweep: match on the config-derived name and common process
    # names to handle cases where the tracked PID is stale or the process
    # has forked children with different names.
    try:
        name = config["name"].lower().replace(" ", "_")
        subprocess.run(["pkill", "-f", name], capture_output=True)
        subprocess.run(["pkill", "-f", "px4"], capture_output=True)
        subprocess.run(["pkill", "-f", "Step_2_Monitoring"], capture_output=True)
        subprocess.run(["pkill", "-f", "live_inference"], capture_output=True)
        subprocess.run(["pkill", "-f", "streamlit"], capture_output=True)
        remove_pid(component_id)
        return True, f"Stopped {config['name']}"
    except:
        pass

    # Graceful fallback: if we have a PID and it is still alive, use the
    # process manager's stop routine (SIGTERM + SIGKILL escalation).
    if not pid or not is_process_running(pid):
        remove_pid(component_id)
        return True, f"{config['name']} was not running"

    success = pm_stop_process(pid)
    if success:
        remove_pid(component_id)
        return True, f"Stopped {config['name']}"
    else:
        return False, f"Failed to stop {config['name']}"


def stop_gazebo():
    """Kill all Gazebo simulation processes.

    Uses aggressive ``killall -9`` targeting well-known Gazebo executables
    (``gzserver``, ``gzclient``) as well as ``pkill`` on ``gz sim`` and
    ``gz_`` to catch any remaining child processes.

    Returns:
        Tuple of ``(success: bool, message: str)``.
    """
    try:
        subprocess.run(["pkill", "-f", "gz sim"], capture_output=True)
        subprocess.run(["killall", "-9", "gzserver"], capture_output=True)
        subprocess.run(["killall", "-9", "gzclient"], capture_output=True)
        subprocess.run(["pkill", "-f", "gz_"], capture_output=True)
        return True, "Gazebo processes killed"
    except:
        return False, "Failed to kill gazebo"


def is_gazebo_running():
    """Check whether any Gazebo ``gz sim`` process is currently alive.

    Returns:
        ``True`` if at least one matching process was found.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-af", "gz sim"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return False
        return bool(result.stdout.strip())
    except Exception as e:
        print(f"DEBUG is_gazebo_running exception: {e}")
        return False


def run_ml_pipeline(pipeline_id):
    """Run an ML pipeline command in a new macOS Terminal window.

    Args:
        pipeline_id: Key into the ML_PIPELINE configuration dict (e.g.
            ``"train"``, ``"evaluate"``).

    Returns:
        Tuple of ``(success: bool, message: str)``.
    """
    config = ML_PIPELINE[pipeline_id]

    cmd = config["command"]
    workdir = str(PROJECT_ROOT)

    try:
        cmd_escaped = cmd.replace("'", "'\\''")

        # macOS-specific: launch the pipeline in a dedicated Terminal window so
        # the user can monitor its progress independently of the TUI.
        script = f'''
        tell application "Terminal"
            activate
            do script "cd '{workdir}' && {cmd_escaped}"
        end tell
        '''
        subprocess.run(["osascript", "-e", script], check=True)

        return True, f"Running {config['name']} in new window"

    except Exception as e:
        return False, f"Error: {e}"


def get_data_info():
    """Gather summary information about available data and model files.

    Scans the configured ``raw``, ``processed``, and ``models`` directories
    and returns counts plus sample file names (limited to the first 10) for
    display in the TUI status panel.

    Returns:
        A dict with keys ``raw_count``, ``raw_files``, ``processed_count``,
        ``processed_files``, ``model_count``, and ``model_files``.
    """
    info = {
        "raw_count": 0,
        "raw_files": [],
        "processed_count": 0,
        "processed_files": [],
        "model_count": 0,
        "model_files": [],
    }

    if RAW_DATA_DIR.exists():
        files = list(RAW_DATA_DIR.glob("*.csv"))
        info["raw_count"] = len(files)
        # Only show the first 10 files to keep the TUI display concise.
        info["raw_files"] = [f.name for f in sorted(files)[:10]]

    if PROCESSED_DATA_DIR.exists():
        files = list(PROCESSED_DATA_DIR.glob("*.csv"))
        info["processed_count"] = len(files)
        info["processed_files"] = [f.name for f in sorted(files)[:10]]

    if MODELS_DIR.exists():
        pkl_files = list(MODELS_DIR.glob("*.pkl"))
        pth_files = list(MODELS_DIR.glob("*.pth"))
        info["model_count"] = len(pkl_files) + len(pth_files)
        info["model_files"] = [f.name for f in sorted(pkl_files) + sorted(pth_files)]

    return info


def start_all_components():
    """Start every component defined in the global COMPONENTS config.

    Iterates all configured component IDs and calls :func:`start_component`
    on each. Aggregates individual results so the caller can report per-
    component status.

    Returns:
        List of ``(component_id, success, message)`` tuples.
    """
    results = []
    for comp_id in COMPONENTS:
        success, msg = start_component(comp_id)
        results.append((comp_id, success, msg))
    return results


def stop_all_components():
    """Stop every component defined in COMPONENTS and kill Gazebo.

    Iterates all configured component IDs and calls :func:`stop_component`
    on each. Also stops Gazebo as a safety measure since it is frequently
    left running when individual components are torn down.

    Returns:
        List of ``(component_id, success, message)`` tuples.
    """
    results = []
    for comp_id in COMPONENTS:
        success, msg = stop_component(comp_id)
        results.append((comp_id, success, msg))
    stop_gazebo()
    return results