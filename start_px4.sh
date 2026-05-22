#!/bin/bash
# PX4 SITL Launcher — GPS Spoofing Project
# Usage: ./start_px4.sh [world_name]   (default: windy)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"
PX4_DIR="$PROJECT_ROOT/Step_1_Simulation/PX4-Autopilot"

# Venv
if [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Dependencies
if ! python3 -c "import em; exit(0 if em.__version__ == '3.3.4' else 1)" 2>/dev/null; then
    python3 -m pip install --quiet "empy==3.3.4"
fi

cd "$PX4_DIR" || { echo "ERROR: PX4 directory not found"; exit 1; }

# Clean stale build cache
if [ -d "build/px4_sitl_default" ]; then
    rm -rf build/px4_sitl_default/CMakeCache.txt build/px4_sitl_default/CMakeFiles
fi

WORLD=${1:-windy}
export PX4_GZ_WORLD=$WORLD
export PYTHON_EXECUTABLE=$(which python3)

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Launching PX4 SITL — World: $WORLD"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo " 1. Wait for → Ready for takeoff!"
echo " 2. Type    → commander arm && commander takeoff"
echo " 3. In QGC  → click map to send waypoint"
echo ""
echo " 4. For GPS spoofer: just run the attack"
echo "    (params auto-configured via MAVLink)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

make px4_sitl gz_x500
