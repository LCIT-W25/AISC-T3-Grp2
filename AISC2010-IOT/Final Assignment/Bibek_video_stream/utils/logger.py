from datetime import datetime

def log_event(event_key):
    with open("event_log.txt", "a") as f:
        f.write(f"{datetime.now().isoformat()} - {event_key}\n")