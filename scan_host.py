import socket
import urllib.request

def check_endpoint(host, port, path=""):
    url = f"http://{host}:{port}{path}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NSA-Discovery"})
        with urllib.request.urlopen(req, timeout=0.8) as resp:
            print(f"[FOUND] {url} -> HTTP {resp.status}")
            return True
    except urllib.error.HTTPError as e:
        print(f"[FOUND] {url} -> HTTP {e.code}")
        return True
    except Exception:
        return False

# Get default gateway IP from route
gateway = None
try:
    with open("/proc/net/route") as f:
        for line in f:
            fields = line.strip().split()
            if fields[1] == '00000000' or fields[0] == 'Iface':
                continue
            if len(fields) >= 3 and fields[1] == '00000000':
                pass
    import subprocess
    out = subprocess.check_output(["ip", "route", "show", "default"], text=True)
    gateway = out.split()[2]
except Exception:
    pass

candidates = ["localhost", "127.0.0.1"]
if gateway:
    candidates.append(gateway)

print(f"Candidate hosts: {candidates}")
for host in candidates:
    # LMStudio port
    check_endpoint(host, 1234, "/v1/models")
    # Ollama port
    check_endpoint(host, 11434, "/api/tags")

print("Scan finished.")
