# game.py

import random

class Game:
    def __init__(self, size=5, num_people=10):
        self.size = size
        self.board = [[' ' for _ in range(size)] for _ in range(size)]
        self.num_people = num_people
        self.characters = ['K', 'D'] + ['P'] * (num_people - 2)  # K: Killer, D: Detective, P: People
        self.game_over = False
        self.winner = None

    def setup_board(self):
        random.shuffle(self.characters)
        for i, char in enumerate(self.characters):
            x, y = divmod(i, self.size)
            self.board[x][y] = char

    def print_board(self):
        for row in self.board:
            print(' '.join(row))
        print()

    def action_kill(self, x, y):
        if self.board[x][y] != 'P':
            return False
        self.board[x][y] = 'X'  # X represents a killed person
        return True

    def action_interrogate(self, x, y):
        if self.board[x][y] == 'K':
            self.game_over = True
            self.winner = 'Detective'
            return True
        return False

    def action_investigate_area(self, x1, y1, x2, y2, role):
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                if self.board[x][y] == role:
                    return True
        return False

    def check_game_over(self):
        if 'D' not in sum(self.board, []):
            self.game_over = True
            self.winner = 'Killer'
