import socket
import sys

HOST = '127.0.0.1'
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8766

print(f"Connecting to {HOST}:{PORT}...")
try:
    with socket.create_connection((HOST, PORT)) as s:
        print(f"Connected to {HOST}:{PORT}")
        while True:
            data = s.recv(4096)
            if not data:
                print("Server closed connection")
                break
            try:
                sys.stdout.write(data.decode('utf-8'))
            except Exception:
                sys.stdout.write(data.decode('utf-8', errors='replace'))
            sys.stdout.flush()
except ConnectionRefusedError:
    print(f"Connection refused on {HOST}:{PORT}. Is the Env server running and enabled?")
except KeyboardInterrupt:
    print("Interrupted by user")
except Exception as e:
    print(f"Client error: {e}")
