import json
import socket
import time

HOST = "0.0.0.0"
PORT = 9000


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    sock.settimeout(1.0)
    print(f"Mock PLC listening on UDP port {PORT}. Press Ctrl+C to stop.")
    try:
        while True:
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            message = json.loads(data.decode("utf-8"))
            classes = [d["class"] for d in message["detections"]]
            stamp = time.strftime("%H:%M:%S")
            print(f"{stamp}  {message['action']:6}  defects={message['count']}  {classes}")
    except KeyboardInterrupt:
        print("\nMock PLC stopped.")


if __name__ == "__main__":
    main()
