import random


def create_board(m, n, x):
    board = [['.' for _ in range(n)] for _ in range(m)]
    positions = [(i, j) for i in range(m) for j in range(n)]
    random.shuffle(positions)

    characters = {}
    for i in range(x):
        position = positions[i]
        if i == 0:
            board[position[0]][position[1]] = 'M'  # Murderer
            characters['M'] = position
        elif i == 1:
            board[position[0]][position[1]] = 'D'  # Detective
            characters['D'] = position
        else:
            board[position[0]][position[1]] = 'P'  # Ordinary person
    return board, characters


def print_board(board):
    for row in board:
        print(' '.join(row))
    print()


def random_action(board, player_type, player_pos):
    if player_type == 'M':
        action_types = ['kill', 'select']
    else:
        action_types = ['accuse', 'select']

    action = random.choice(action_types)

    if action == 'kill' or action == 'accuse':
        possible_targets = [(i, j) for i in range(len(board)) for j in range(len(board[0])) if
                            board[i][j] in ['P', 'D', 'M'] and (i, j) != player_pos]
        if possible_targets:
            target = random.choice(possible_targets)
            return action, target
    elif action == 'select':
        m, n = len(board), len(board[0])
        x1, y1 = random.randint(0, m - 1), random.randint(0, n - 1)
        x2, y2 = random.randint(x1, min(m - 1, x1 + m // 2)), random.randint(y1, min(n - 1, y1 + n // 2))
        return 'select', (x1, y1, x2, y2)

    return 'pass', ()


def perform_action(board, action, coords, player_type, characters):
    info = ""
    if action == 'kill':
        x, y = coords
        if 0 <= x < len(board) and 0 <= y < len(board[0]) and board[x][y] in ['P', 'D']:
            killed = board[x][y] == 'D'
            board[x][y] = '.'
            if killed:
                characters.pop('D', None)
                info = "Detective was killed."
            else:
                info = f"Person at ({x}, {y}) was killed."
            return 'killed' if killed else 'continue', info
    elif action == 'accuse':
        x, y = coords
        accused = board[x][y] == 'M'
        if accused:
            characters.pop('M', None)
            info = "Murderer was accused correctly."
            return 'accused', info
        else:
            info = "Accused person was not the murderer."
            return 'continue', info
    elif action == 'select':
        x1, y1, x2, y2 = coords
        found = False
        for i in range(max(0, x1), min(len(board), x2 + 1)):
            for j in range(max(0, y1), min(len(board[0]), y2 + 1)):
                if (player_type == 'M' and board[i][j] == 'D') or (player_type == 'D' and board[i][j] == 'M'):
                    found = True
                    break
            if found:
                break
        info = f"Detective/Murderer {'found' if found else 'not found'} in selected area."
        return 'found' if found else 'continue', info

    return 'continue', info


def count_people(board):
    return sum(row.count('P') + row.count('M') + row.count('D') for row in board)


def main(m, n, x):
    board, characters = create_board(m, n, x)
    total_people = count_people(board)
    turn = 'M'  # M: Murderer, D: Detective
    game_over = False

    while not game_over and 'M' in characters and 'D' in characters:
        print_board(board)
        print(f"{turn}'s turn.")

        action, coords = random_action(board, turn, characters[turn])
        print(f"Action: {action}, Coords: {coords}")
        result, info = perform_action(board, action, coords, turn, characters)
        print(info)

        if (turn == 'M' and result == 'killed') or (turn == 'D' and result == 'accused'):
            game_over = True
            score = (total_people - count_people(board)) / total_people if turn == 'M' else count_people(
                board) / total_people
            print(f"{turn} wins!")

        turn = 'D' if turn == 'M' else 'M'

    if not game_over:
        print("Game ended with no conclusive result.")

    print(f"Game over. Score: {score:.2f}")


if __name__ == "__main__":
    main(5, 5, 10)  # Example: 5x5 board with 10 people
