# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
import struct

from collections import defaultdict


COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None, savepath=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        if savepath:
            fig.savefig(savepath)
            plt.close()
        else:
            plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)


########################################################################
#                       given environment above                        #
########################################################################
class Pattern:
    def __init__(self, pattern, iso=8):
        self.pattern = pattern
        self.iso = iso
        self.weights = None
        self.isom = self._create_isomorphic_patterns()

    def _create_isomorphic_patterns(self):
        isom = []
        for i in range(self.iso):
            idx = self._rotate_mirror_pattern([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], i)
            patt = [idx[p] for p in self.pattern]
            isom.append(patt)
        return isom

    def _rotate_mirror_pattern(self, base, rot):
        board = np.array(base, dtype=int).reshape(4,4)
        if rot >= 4:
            board = np.fliplr(board)
        board = np.rot90(board, rot % 4)
        return board.flatten().tolist()

    def load_weights(self, weights):
        self.weights = weights

    def estimate(self, board):
        total = 0.0
        for iso in self.isom:
            index = self._get_index(iso, board)
            total += self.weights[index]
        return total

    def _get_index(self, pattern, board):
        index = 0
        for i, pos in enumerate(pattern):
            tile = board[pos//4][pos%4]
            if tile == 0: 
                val = 0
            else:
                val = int(np.log2(tile))
            index |= (val & 0xF) << (4 * i)
        return index

class CPPModel:
    def __init__(self, bin_path):
        self.patterns = []
        self._load_binary(bin_path)

    def _load_binary(self, path):
        with open(path, 'rb') as f:
            num_features = struct.unpack('Q', f.read(8))[0]
            
            for _ in range(num_features):
                # Read feature name
                name_len = struct.unpack('I', f.read(4))[0]
                name = f.read(name_len).decode('utf-8')
                
                # Parse pattern from name (e.g., "4-tuple pattern 0123")
                pattern = [int(c, 16) for c in name.split()[-1]]
                
                # Create pattern and load weights
                p = Pattern(pattern)
                size = struct.unpack('Q', f.read(8))[0]
                weights = struct.unpack(f'{size}f', f.read(4*size))
                p.load_weights(weights)
                self.patterns.append(p)

    def estimate(self, board):
        return sum(p.estimate(board) for p in self.patterns)

model = CPPModel("2048.bin")

class Node:
    def __init__(self, action=None, parent=None):
        self.action = action      # 該節點代表所採取的行動（例如：上、下、左、右）
        self.parent = parent      # 父節點
        self.children = []        # 子節點列表
        self.visits = 0           # 被訪問次數
        self.value = 0.0          # 累計評估值

def select_child(node: Node) -> Node:
    """
    從給定的節點中選擇一個子節點進行模擬：
      - 若有未被訪問過的子節點，則隨機回傳其中之一。
      - 否則根據 UCB（上限信心值）公式選出最佳子節點。
    """
    if not node.children:
        return None

    # 優先選擇尚未被訪問過的子節點
    unvisited = [child for child in node.children if child.visits == 0]
    if unvisited:
        return random.choice(unvisited)

    # 設定探索參數 C
    C = 1.0
    best_score = -float('inf')
    best_child = None
    # 使用 UCB1 公式選出最佳子節點
    for child in node.children:
        if child.visits > 0:
            ucb = (child.value / child.visits) + C * math.sqrt(math.log(node.visits) / child.visits)
            if ucb > best_score:
                best_score = ucb
                best_child = child
    return best_child

def board_to_cpp_format(py_board):
    """
    將 Python 二維陣列型的遊戲盤轉換成 C++ 風格的 64 位元整數格式。
    每個棋盤位置以 4 位元表示，依序組合成 64 位的整數。
    """
    cpp_board = 0
    for i in range(4):
        row = 0
        for j in range(4):
            tile = py_board[i][j]
            # 若該位置為空（0）則儲存 0，否則取對數得到指數值
            val = 0 if tile == 0 else int(np.log2(tile))
            row |= (val & 0xF) << (j * 4)
        cpp_board |= row << (i * 16)
    return cpp_board

def get_action(state, score):
    """
    利用蒙地卡羅樹搜尋 (MCTS) 與 n-tuple 模型估值，決定遊戲 2048 中的最佳下一步行動。

    參數:
        state: 目前的遊戲盤狀態（4x4 二維陣列）。
        score: 當前分數。

    回傳:
        最佳行動代碼（0: 上, 1: 下, 2: 左, 3: 右）。
    """
    # 初始化遊戲環境，此處假設 Game2048Env 在其他地方已定義
    env = Game2048Env()
    env.board = np.array(state)
    env.score = score

    # 選取所有合法的移動方向
    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    if not legal_moves:
        return 0  # 理論上不應該會遇到無合法移動的情況

    # 建立根節點，並為每一個合法行動建立子節點
    root = Node()
    for action in legal_moves:
        child = Node(action=action, parent=root)
        root.children.append(child)

    # 模擬次數可根據合法移動數目來設定（此處僅進行固定次數模擬）
    num_iterations = len(legal_moves)
    for _ in range(num_iterations):
        # 從根節點中選擇一個子節點
        child = select_child(root)
        # 複製環境以避免改變原始狀態
        temp_env = copy.deepcopy(env)

        # 根據選擇的行動進行模擬移動
        if child.action == 0:
            temp_env.move_up()
        elif child.action == 1:
            temp_env.move_down()
        elif child.action == 2:
            temp_env.move_left()
        elif child.action == 3:
            temp_env.move_right()

        # 使用 n-tuple 模型估計盤面價值，並結合移動前後分數的差異進行評估
        estimated_value = model.estimate(temp_env.board) + (temp_env.score - score)

        # 模擬後隨機新增一個新瓦片（90% 為 2，10% 為 4）
        empty_cells = list(zip(*np.where(temp_env.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            tile_value = 2 if random.random() < 0.9 else 4
            temp_env.board[x, y] = tile_value

        # 若模擬後遊戲結束，賦予一個大幅負分作為懲罰
        if temp_env.is_game_over():
            estimated_value = -50000

        # 更新節點統計數據
        child.visits += 1
        child.value += estimated_value
        root.visits += 1

    # 從子節點中選出總價值最大的作為最佳行動
    best_child = max(root.children, key=lambda c: c.value)
    return best_child.action

# def run_episode(env, render=False, render_interval=50):
#          state = env.reset()
#          total_reward = 0
#          steps = 0
#          max_tile = 0
         
#          while True:
#              if render and (steps % render_interval == 0):
#                  env.render(action=None, savepath=f"frame_{steps:04d}.png")
             
#              action = get_action(state, env.score)
#              next_state, reward, done, _ = env.step(action)
             
#              total_reward += reward
#              steps += 1
#              max_tile = max(max_tile, np.max(next_state))
#              state = next_state
#              if done:
#                  break
                 
#          return env.score, max_tile, steps
# if __name__ == "__main__":
#      # main()
#     import argparse
#     import numpy as np
#     from tqdm import tqdm
#     from collections import defaultdict
#     import subprocess
#     url = "https://www.dropbox.com/scl/fi/2e5zpv08d93vt0p9xzrig/2048.bin?rlkey=urbv8m20unlrflnf5gyruanln&st=0z7n1itc&dl=0"
#     output_file = "2048.bin"

#     subprocess.run(["curl", "-L", "-o", output_file, url], check=True)
#     model = CPPModel("2048.bin")
#     from test import *
#     random.seed(0)
#     for i in range(10):
#         env = Game2048Env()
#         env.reset()
#         print(run_episode(env))
 