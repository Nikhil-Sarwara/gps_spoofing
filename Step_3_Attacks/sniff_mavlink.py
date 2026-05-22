import time
from pymavlink import mavutil

def main():
    print("[*] Sniffing MAVLink traffic on 14540...")
    conn = mavutil.mavlink_connection("udp:127.0.0.1:14540")
    
    start_time = time.time()
    counts = {}
    
    while time.time() - start_time < 5:
        msg = conn.recv_match(blocking=True, timeout=1.0)
        if msg:
            mtype = msg.get_type()
            counts[mtype] = counts.get(mtype, 0) + 1
            if mtype in ['GPS_RAW_INT', 'HIL_GPS', 'GPS_INPUT']:
                print(f"  [FOUND] {msg}")
                
    print("\nMessage Summary (5s):")
    for t, c in counts.items():
        print(f"  {t}: {c}")

if __name__ == "__main__":
    main()
