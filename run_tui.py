#!/usr/bin/env python3
"""
GPS Spoofing Detection System - TUI Management Interface
Main entry point for the TUI application
"""

import os
import sys
import subprocess
from pathlib import Path

# Check if virtual environment is activated
venv_active = False
if hasattr(sys, "real_prefix"):
    venv_active = True
elif hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix:
    venv_active = True
elif os.environ.get("VIRTUAL_ENV") is not None:
    venv_active = True

if not venv_active:
    print("ERROR: Virtual environment not activated!")
    print("")
    print("Please activate the virtual environment first:")
    print("  source venv/bin/activate")
    print("")
    print("If you do not have a venv yet, create one with:")
    print("  python3 -m venv venv")
    sys.exit(1)

# Check and install dependencies
try:
    from textual.app import App
except ImportError as e:
    print("ERROR: Missing dependency: {}".format(e))
    print("")
    print("Installing textual and rich...")
    subprocess.run([sys.executable, "-m", "pip", "install", "textual", "rich"])
    print("Dependencies installed. Please restart the script.")
    sys.exit(0)


def main():
    """Main entry point."""
    # Ensure project root is on sys.path for package imports
    root = Path(__file__).parent.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    
    # Import as a package member to support relative imports in app.py
    from Step_6_UI.tui_console.app import run_app
    run_app()


if __name__ == "__main__":
    main()
