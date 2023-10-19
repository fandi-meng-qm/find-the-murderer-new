from murder_game_core import MurderGame, MurderParams, MurderState, get_init_states, MurderObserver
import subprocess

params = MurderParams(1, 3, 3)
game = MurderGame(game_params=params)


class Node:
    def __init__(self, state, children=None):
        self.state = state
        obs = MurderGame.make_py_observer(game)
        if type(state) is list:
            self.str = str(state)
        else:
            if state.step == 0:
                self.str = 'Root'
            else:
                state_str = MurderObserver.string_from(obs, state, state.to_play())
                state_str=state_str.replace('],', ']\n')
                self.str = state_str
        self.children = children if children is not None else []


def create_tree(parent_node, model):
    if parent_node.state.is_chance_node():
        chance_states = get_init_states(parent_node.state.params)
        for i in chance_states:
            child_node = Node(i)
            parent_node.children.append(child_node)
            create_tree(child_node, model)
    else:
        if parent_node.state.is_terminal():
            score = parent_node.state.returns()
            child_node = Node(score)
            parent_node.children.append(child_node)
        else:
            player = parent_node.state.to_play()
            actions = parent_node.state._legal_actions(player)
            for a in actions:
                child_node = Node(parent_node.state.child(a))
                child_node.str = 'action' + str(a) + '->observation''\n' + child_node.str
                parent_node.children.append(child_node)
                create_tree(child_node, model)
    return parent_node


def tree_to_dot(root):
    dot = ["digraph GameTree {"]

    nodes_to_process = [root]

    while nodes_to_process:
        current = nodes_to_process.pop()
        for child in current.children:
            dot.append(f'    "{current.str}" -> "{child.str}";')
            nodes_to_process.append(child)

    dot.append("}")
    return "\n".join(dot)


parent_node = Node(game.new_initial_state())
game_tree = create_tree(parent_node, game)

# print(tree_to_dot(game_tree))

dot_content = tree_to_dot(game_tree)
with open("game_tree.dot", "w") as file:
    file.write(dot_content)

subprocess.run(["dot", "-Tpng", "game_tree.dot", "-o", "game_tree.png"])
