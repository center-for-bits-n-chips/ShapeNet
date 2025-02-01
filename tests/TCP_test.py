import socket
import struct
import threading

def handle_client(stop_event, conn, addr):
    print(f"Connected by {addr}")
    conn.settimeout(1.0)  # Set timeout to prevent blocking indefinitely
    try:
        while not stop_event.is_set():
            # Receive data from LabVIEW
            try:
                data = conn.recv(1024)
                if data:
                    # Process received data (assuming big-endian 4-byte float)
                    while len(data) >= 8:
                        num = struct.unpack('>d', data[:8])[0]
                        print(f"Received from LabVIEW: {num}")
                        data = data[8:]
                else:
                    print(f"Client {addr} closed the connection.")
                    break  # Connection closed by client
            except socket.timeout:
                pass  # No data received; proceed to sending
            except (ConnectionResetError, ConnectionAbortedError):
                print(f"Connection with {addr} was reset during receive.")
                break
            except Exception as e:
                print(f"Error receiving data from {addr}: {e}")
                break

            # Send data to LabVIEW
            num_to_send = 2 * num  # Example number
            data_to_send = struct.pack('>d', num_to_send)
            try:
                conn.sendall(data_to_send)
                print(f"Sent to LabVIEW: {num_to_send}")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                print(f"Connection with {addr} was closed during send.")
                break
            except Exception as e:
                print(f"Error sending data to {addr}: {e}")
                break

    finally:
        conn.close()
        print(f"Connection with {addr} closed.")

def start_server(host, port):
    # Create a stop event
    stop_event = threading.Event()
    threads = []
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        s.settimeout(1.0)  # Set timeout on socket operations
        print(f"Server listening on port {port}")
        try:
            while not stop_event.is_set():
                try:
                    conn, addr = s.accept()
                    print(f"Accepted connection from {addr}")
                    thread = threading.Thread(target=handle_client, args=(stop_event, conn, addr))
                    thread.start()
                    threads.append(thread)
                    # If you want to handle only one connection, you can break here
                    # break
                except socket.timeout:
                    pass  # No connection attempt; check stop_event
        except KeyboardInterrupt:
            print("Main thread received KeyboardInterrupt. Stopping server...")
            stop_event.set()
        finally:
            print("Server is shutting down.")
            s.close()
            # Wait for all threads to finish
            for t in threads:
                t.join()
            print("All client connections have been closed.")

def main():
    HOST = '127.0.0.1'  # Listen on specified network interface
    PORT = 9999         # Arbitrary non-privileged port
    start_server(HOST, PORT)

if __name__ == '__main__':
    main()
