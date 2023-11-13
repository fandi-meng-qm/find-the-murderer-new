# server.py

import socket
import threading
from game import Game

class GameServer:
    def __init__(self, host='127.0.0.1', port=12345):
        self.game = Game()
        self.game.setup_board()
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clients = []
        self.roles = ['Killer', 'Detective']

    def handle_client(self, client, role):
        client.send(f"You are {role}".encode())
        while not self.game.game_over:
            data = client.recv(1024).decode()
            response = self.process_command(data, role)
            client.send(response.encode())
            if self.game.game_over:
                message = "Game Over! " + ("You win!" if self.game.winner == role else "You lose!")
                client.send(message.encode())
        client.close()

    def process_command(self, data, role):
        commands = data.split()
        if len(commands) < 3:
            return "Invalid command. Try again."

        action, x, y = commands[0], int(commands[1]), int(commands[2])
        if action == "kill" and role == "Killer":
            if self.game.action_kill(x, y):
                return "You killed at position ({}, {}).".format(x, y)
            else:
                return "Invalid kill action."
        elif action == "interrogate" and role == "Detective":
            if self.game.action_interrogate(x, y):
                return "You found the killer!"
            else:
                return "Interrogation at ({}, {}) found nothing.".format(x, y)
        else:
            return "Invalid action or role."

    def start(self):
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(2)
        print("Game Server Started.")

        for _ in range(2):
            client, addr = self.server_socket.accept()
            self.clients.append(client)
            role = self.roles.pop(0)
            threading.Thread(target=self.handle_client, args=(client, role)).start()

def main():
    server = GameServer()
    server.start()

if __name__ == '__main__':
    main()
