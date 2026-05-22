# GPS Spoofing Detection System

A machine learning-based GPS anomaly/spoofing detection system for PX4 drones, organized following the principles of "divide and conquer" problem decomposition (Jones et al., CHI 2005).

## Task-Oriented Project Structure

The project is decomposed into 8 logical steps, each representing a critical component of the research and implementation workflow:

```
gps_spoofing/
├── Step_1_Simulation/   # Simulation Environment (PX4 SITL, Gazebo)
├── Step_2_Monitoring/   # Telemetry Data Acquisition (MAVLink collector)
├── Step_3_Attacks/      # Threat Simulation (Attack vectors & sniffer)
├── Step_4_Detection/    # Detection Intelligence (ML Pipeline & Inference)
├── Step_5_Data/         # Information Vault (Raw logs, Processed data, Models)
├── Step_6_UI/           # Visualization & Control (Streamlit & TUI Console)
├── Step_7_Research/     # Academic Output (Thesis, Papers, LaTeX)
└── Step_8_Archive/      # Project History (Legacy logs, test outputs)
```

## Quick Start (Project Cockpit)

Launch the integrated management console to orchestrate all components:

```bash
# Launch the main control interface
python3 run_tui.py
```

### Main Entry Points (Root)

- `run_tui.py`: The management console (Dashboards & ML Pipeline).
- `start_px4.sh`: Direct launcher for the PX4 SITL simulation.
- `gps_spoofer.py`: Manual attack injector for testing detection.

## Core Innovation: Physical Sanity Check

This system distinguishes between environmental noise (e.g., wind) and active spoofing attacks by correlating GPS motion with IMU-detected physical acceleration. In a spoofing attack, the GPS coordinates shift without corresponding physical motion detected by the accelerometers and gyroscopes.

## ML Pipeline

The automated pipeline in `Step_4_Detection` handles the entire data lifecycle:
1. **Cleaning**: Filtering invalid data and handling EKF resets.
2. **Labeling**: Applying 8 automated heuristics to identify anomalies.
3. **Training**: Generating Random Forest and 1D CNN models optimized for different terrains (Flat, Mountain, Sea).

---
*Organized for digital findability and project decomposition.*
