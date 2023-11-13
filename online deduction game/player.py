# client.py

import socket

def main():
    host = '127.0.0.1'  # 服务器的IP地址
    port = 12345

    client_socket = socket.socket()
    client_socket.connect((host, port))

    role = client_socket.recv(1024).decode()
    print(role)

    while True:
        data = client_socket.recv(1024).decode()
        if "Game Over" in data:
            print(data)
            break
        print(data)
        action = input("Enter your action: ")
        client_socket.send(action.encode())

    client_socket.close()

if __name__ == '__main__':
    main()
