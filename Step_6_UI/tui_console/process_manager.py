# TUI Process Manager - Process management functions
import os
import signal
import subprocess
import time
from pathlib import Path
from .config import COMPONENTS


def is_process_running(pid):
    """Check if a process with given PID is running."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def find_pid_by_name(component):
    """Find PID by searching process list for component command."""
    config = COMPONENTS.get(component)
    if not config:
        return None
    
    name = config["name"].lower()
    search_terms = {
        "dashboard": ["streamlit", "Step_6_UI/web_ui/app.py"],
        "gps_monitor": ["Step_2_Monitoring.main"],
        "live_inference": ["Step_4_Detection.live_inference"],
        "px4_sim": ["bin/px4", "start_px4.sh"],
    }
    
    terms = search_terms.get(component, [name])
    try:
        result = subprocess.run(
            ["pgrep", "-af", "|".join(terms)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            if pids and pids[0]:
                return int(pids[0].split()[0])
    except:
        pass
    return None


def get_pid(component):
    """Read PID from file, or find by process name if not found."""
    pid_file = COMPONENTS[component]["pid_file"]
    if pid_file.exists():
        try:
            saved_pid = int(pid_file.read_text().strip())
            if saved_pid and is_process_running(saved_pid):
                return saved_pid
        except (ValueError, IOError):
            pass
    
    return find_pid_by_name(component)


def save_pid(component, pid):
    """Save PID to file."""
    pid_file = COMPONENTS[component]["pid_file"]
    pid_file.write_text(str(pid))


def remove_pid(component):
    """Remove PID file."""
    pid_file = COMPONENTS[component]["pid_file"]
    if pid_file.exists():
        pid_file.unlink()


def get_all_pids():
    """Get PIDs for all components."""
    return {comp_id: get_pid(comp_id) for comp_id in COMPONENTS}


def get_running_components():
    """Get list of currently running components."""
    running = []
    for comp_id in COMPONENTS:
        pid = get_pid(comp_id)
        if pid and is_process_running(pid):
            running.append(comp_id)
    return running


def stop_process(pid):
    """Stop a process by PID."""
    if not pid or not is_process_running(pid):
        return False
    
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        else:
            os.kill(pid, signal.CTRL_C_EVENT)
        
        time.sleep(1)
        
        if is_process_running(pid):
            if os.name != "nt":
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGTERM)
        
        return True
    except Exception:
        return False


def stop_all_processes():
    """Stop all running component processes."""
    for comp_id in COMPONENTS:
        pid = get_pid(comp_id)
        if pid and is_process_running(pid):
            stop_process(pid)
            remove_pid(comp_id)