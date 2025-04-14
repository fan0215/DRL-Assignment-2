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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import os
import heapq
from collections import defaultdict
import time

# 顏色映射字典（缺失）
COLOR_MAP = {
    0: "#cdc1b4",
    2: "#eee4da",
    4: "#ede0c8",
    8: "#f2b179",
    16: "#f59563",
    32: "#f67c5f",
    64: "#f65e3b",
    128: "#edcf72",
    256: "#edcc61",
    512: "#edc850",
    1024: "#edc53f",
    2048: "#edc22e",
}

TEXT_COLOR = {
    0: "#776e65",
    2: "#776e65",
    4: "#776e65",
    8: "#f9f6f2",
    16: "#f9f6f2",
    32: "#f9f6f2",
    64: "#f9f6f2",
    128: "#f9f6f2",
    256: "#f9f6f2",
    512: "#f9f6f2",
    1024: "#f9f6f2",
    2048: "#f9f6f2",
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

    def render(self, mode="human", action=None):
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

# 定義DQN網絡
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(16 + 7, 128)  # 16個格子狀態 + 7個特徵
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 4)  # 4個動作
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 經驗回放記憶體
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Experience(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# 特徵提取函數
def extract_features(board, score):
    features = []
    
    # 特徵1: 空格數
    empty_cells = (board == 0).sum()
    features.append(empty_cells / 16.0)  # 歸一化
    
    # 特徵2: 最大磚塊值（取對數）
    max_tile = np.max(board)
    features.append(np.log2(max_tile + 1) / 11.0)  # 假設最大磚塊為2048，log2(2048)=11
    
    # 特徵3: 分數（歸一化）
    features.append(np.log2(score + 1) / 20.0)  # 粗略歸一化
    
    # 特徵4: 單調性（磚塊是否遞增/遞減排列）
    def monotonicity(board):
        mono_left_right = 0
        mono_up_down = 0
        
        for i in range(4):
            # 檢查行的單調性
            for j in range(3):
                if board[i, j] != 0 and board[i, j+1] != 0:
                    if board[i, j] >= board[i, j+1]:
                        mono_left_right += 1
            
            # 檢查列的單調性
            for j in range(3):
                if board[j, i] != 0 and board[j+1, i] != 0:
                    if board[j, i] >= board[j+1, i]:
                        mono_up_down += 1
        
        return (mono_left_right + mono_up_down) / 24.0  # 最大可能值為24
    
    features.append(monotonicity(board))
    
    # 特徵5: 平滑度（相鄰磚塊差異）
    def smoothness(board):
        smoothness_val = 0
        for i in range(4):
            for j in range(4):
                if board[i, j] != 0:
                    # 檢查右側
                    if j < 3 and board[i, j+1] != 0:
                        smoothness_val += abs(np.log2(board[i, j]) - np.log2(board[i, j+1]))
                    # 檢查下方
                    if i < 3 and board[i+1, j] != 0:
                        smoothness_val += abs(np.log2(board[i, j]) - np.log2(board[i+1, j]))
        
        return 1.0 / (1.0 + smoothness_val / 20.0)  # 轉換為0-1範圍
    
    features.append(smoothness(board))
    
    # 特徵6: 合併可能性
    def merge_possibility(board):
        merge_count = 0
        for i in range(4):
            for j in range(3):
                if board[i, j] != 0 and board[i, j] == board[i, j+1]:
                    merge_count += 1
        
        for j in range(4):
            for i in range(3):
                if board[i, j] != 0 and board[i, j] == board[i+1, j]:
                    merge_count += 1
        
        return merge_count / 12.0  # 最大可能值為12
    
    features.append(merge_possibility(board))
    
    # 特徵7: 角落磚塊權重
    corner_weight = 0
    corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
    for i, j in corners:
        if board[i, j] == max_tile:
            corner_weight = 1.0
            break
    
    features.append(corner_weight)
    
    return np.array(features, dtype=np.float32)
    
# 預處理狀態為DQN輸入
def preprocess_state(board, score):
    # 將遊戲板變平為一維數組
    flat_board = board.flatten() / 2048.0  # 歸一化
    features = extract_features(board, score)
    # 將板和特徵結合為一個輸入
    return np.concatenate([flat_board, features])

# DQN代理
class DQNAgent:
    def __init__(self, state_size=23, action_size=4, batch_size=64, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_size=10000,
                 load_model=False):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(memory_size)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00025)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        
        self.steps_done = 0
        
        # 嘗試加載保存的模型
        if load_model and os.path.exists('2048_dqn_model.pth'):
            self.load_model('2048_dqn_model.pth')
            print("加載預訓練模型")
    
    def select_action(self, state, env):
        # 轉換狀態為張量
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 以epsilon概率隨機選擇動作
        if random.random() < self.epsilon:
            # 只選擇合法動作
            legal_actions = [a for a in range(self.action_size) if env.is_move_legal(a)]
            if not legal_actions:  # 如果沒有合法動作
                return random.randint(0, self.action_size - 1)
            return random.choice(legal_actions)
        else:
            with torch.no_grad():
                # 獲取所有動作的Q值
                q_values = self.policy_net(state)
                
                # 創建一個極小的Q值掩碼給不合法動作
                action_mask = torch.tensor([
                    [-float('inf') if not env.is_move_legal(a) else 0 for a in range(self.action_size)]
                ], device=self.device)
                
                # 將掩碼應用到Q值
                masked_q_values = q_values + action_mask
                
                # 選擇Q值最高的合法動作
                return masked_q_values.max(1)[1].item()
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
            
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # 轉換為張量
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(np.array(batch.done)).to(self.device)
        
        # 計算當前Q值
        current_q = self.policy_net(state_batch).gather(1, action_batch)
        
        # 計算目標Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
        target_q = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
        
        # 計算損失
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        
        # 梯度下降
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # 更新學習率
        self.scheduler.step()
        
        # 衰減探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, filename='2048_dqn_model.pth'):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon
        }, filename)
    
    def load_model(self, filename='2048_dqn_model.pth'):
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.epsilon = checkpoint['epsilon']

# 訓練函數
def train_agent(episodes=1000, update_target_every=10, save_every=100):
    env = Game2048Env()
    agent = DQNAgent(load_model=True)
    scores = []
    max_score = 0
    
    for e in range(episodes):
        state = env.reset()
        score = 0
        done = False
        
        while not done:
            # 預處理狀態
            state_processed = preprocess_state(state, score)
            
            # 選擇動作
            action = agent.select_action(state_processed, env)
            
            # 執行動作
            next_state, next_score, done, _ = env.step(action)
            
            # 計算獎勵
            reward = next_score - score
            
            # 如果動作無效，給予懲罰
            if not env.last_move_valid:
                reward = -10
            
            # 預處理下一個狀態
            next_state_processed = preprocess_state(next_state, next_score)
            
            # 保存經驗
            agent.memory.push(state_processed, action, next_state_processed, reward, done)
            
            # 優化模型
            agent.optimize_model()
            
            # 更新狀態和分數
            state = next_state
            score = next_score
        
        scores.append(score)
        
        # 更新目標網絡
        if e % update_target_every == 0:
            agent.update_target_network()
        
        # 保存模型
        if e % save_every == 0 or score > max_score:
            agent.save_model()
            if score > max_score:
                max_score = score
        
        # 打印進度
        if e % 10 == 0:
            print(f"Episode: {e}, Score: {score}, Epsilon: {agent.epsilon:.4f}, Average Score (last 10): {np.mean(scores[-10:]):.2f}")
    
    return agent, scores

# 高級啟發式評估函數
class Heuristics:
    @staticmethod
    def empty_tiles(board):
        """計算空格數"""
        return np.sum(board == 0)
    
    @staticmethod
    def max_tile(board):
        """返回最大磚塊值"""
        return np.max(board)
    
    @staticmethod
    def smoothness(board):
        """衡量相鄰磚塊的差異，值越小越好"""
        smoothness = 0
        for i in range(4):
            for j in range(4):
                if board[i, j] != 0:
                    # 檢查相鄰的磚塊
                    for di, dj in [(0, 1), (1, 0)]:  # 右和下
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 4 and 0 <= nj < 4 and board[ni, nj] != 0:
                            smoothness -= abs(math.log2(board[i, j]) - math.log2(board[ni, nj]))
        return smoothness
    
    @staticmethod
    def monotonicity(board):
        """衡量磚塊是否按照單調順序排列"""
        # 計算每個方向的單調性（左到右，上到下）
        monotonicity_scores = [0, 0, 0, 0]
        
        # 左到右
        for i in range(4):
            for j in range(3):
                if board[i, j] != 0 and board[i, j+1] != 0:
                    if board[i, j] > board[i, j+1]:
                        monotonicity_scores[0] += math.log2(board[i, j]) - math.log2(board[i, j+1])
                    else:
                        monotonicity_scores[1] += math.log2(board[i, j+1]) - math.log2(board[i, j])
        
        # 上到下
        for j in range(4):
            for i in range(3):
                if board[i, j] != 0 and board[i+1, j] != 0:
                    if board[i, j] > board[i+1, j]:
                        monotonicity_scores[2] += math.log2(board[i, j]) - math.log2(board[i+1, j])
                    else:
                        monotonicity_scores[3] += math.log2(board[i+1, j]) - math.log2(board[i, j])
        
        # 返回單調性最強的兩個方向的和
        return max(monotonicity_scores[0], monotonicity_scores[1]) + max(monotonicity_scores[2], monotonicity_scores[3])
    
    @staticmethod
    def merge_potential(board):
        """衡量可能合併的磚塊數量"""
        merge_count = 0
        # 水平方向
        for i in range(4):
            for j in range(3):
                if board[i, j] != 0 and board[i, j] == board[i, j+1]:
                    merge_count += 1
        
        # 垂直方向
        for j in range(4):
            for i in range(3):
                if board[i, j] != 0 and board[i, j] == board[i+1, j]:
                    merge_count += 1
        
        return merge_count
    
    @staticmethod
    def corner_max(board):
        """檢查最大磚塊是否在角落"""
        max_val = np.max(board)
        corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
        
        for i, j in corners:
            if board[i, j] == max_val:
                return max_val
        return 0
    
    @staticmethod
    def snake_pattern(board):
        """獎勵蛇形模式的磚塊排列，這是2048的常用策略"""
        snake_weights = np.array([
            [2**15, 2**14, 2**13, 2**12],
            [2**8,  2**9,  2**10, 2**11],
            [2**7,  2**6,  2**5,  2**4],
            [2**0,  2**1,  2**2,  2**3]
        ])
        
        snake_weights2 = np.array([
            [2**15, 2**14, 2**13, 2**12],
            [2**8,  2**9,  2**10, 2**11],
            [2**7,  2**6,  2**5,  2**4],
            [2**0,  2**1,  2**2,  2**3]
        ])
        
        reverse_snake_weights = np.fliplr(snake_weights)
        reverse_snake_weights2 = np.flipud(snake_weights)
        
        pattern1 = np.sum(board * snake_weights)
        pattern2 = np.sum(board * reverse_snake_weights)
        pattern3 = np.sum(board * snake_weights2)
        pattern4 = np.sum(board * reverse_snake_weights2)
        
        return max(pattern1, pattern2, pattern3, pattern4)
    
    @staticmethod
    def edge_tiles(board):
        """獎勵邊緣的高值磚塊"""
        edge_sum = 0
        # 上下邊緣
        for j in range(4):
            edge_sum += board[0, j] + board[3, j]
        
        # 左右邊緣
        for i in range(1, 3):
            edge_sum += board[i, 0] + board[i, 3]
        
        return edge_sum
    
    @staticmethod
    def gradient_score(board):
        """評估磚塊是否形成梯度（從角落遞減）"""
        # 嘗試四個角落
        gradients = []
        corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
        
        for corner_i, corner_j in corners:
            gradient = 0
            for i in range(4):
                for j in range(4):
                    if board[i, j] != 0:
                        # 計算到角落的曼哈頓距離
                        distance = abs(i - corner_i) + abs(j - corner_j)
                        # 磚塊值應該隨距離遞減
                        gradient += board[i, j] * (4 - distance)
            gradients.append(gradient)
        
        return max(gradients)
    
    @staticmethod
    def evaluate(board, score):
        """結合多個啟發式函數評估棋盤狀態"""
        if np.max(board) < 2:  # 防止數學錯誤
            return 0
            
        # 權重
        w_empty = 10.0
        w_max_tile = 1.0
        w_smoothness = 0.1
        w_monotonicity = 1.0
        w_merge = 1.0
        w_corner = 2.0
        w_snake = 1.0
        w_edge = 0.5
        w_gradient = 1.0
        
        # 計算各個指標
        empty = Heuristics.empty_tiles(board)
        max_tile = math.log2(Heuristics.max_tile(board))
        smoothness = Heuristics.smoothness(board)
        monotonicity = Heuristics.monotonicity(board)
        merge = Heuristics.merge_potential(board)
        corner = math.log2(Heuristics.corner_max(board) + 1)
        snake = math.log2(Heuristics.snake_pattern(board) + 1)
        edge = math.log2(Heuristics.edge_tiles(board) + 1)
        gradient = math.log2(Heuristics.gradient_score(board) + 1)
        
        # 結合評分
        total_score = (w_empty * empty +
                      w_max_tile * max_tile +
                      w_smoothness * smoothness +
                      w_monotonicity * monotonicity +
                      w_merge * merge +
                      w_corner * corner +
                      w_snake * snake +
                      w_edge * edge +
                      w_gradient * gradient)
        
        return total_score

# 實現期望最大搜索算法
class ExpectimaxAgent:
    def __init__(self, depth=3):
        self.depth = depth
        self.move_cache = {}  # 快取搜索結果
        
    def get_action(self, board, score):
        """選擇最佳動作"""
        # 動作：0: up, 1: down, 2: left, 3: right
        
        # 壓縮棋盤狀態成字串作為快取鍵
        board_key = board.tobytes()
        
        # 檢查快取
        if board_key in self.move_cache:
            return self.move_cache[board_key]
        
        best_score = -float('inf')
        best_action = -1
        
        # 創建遊戲環境副本用於模擬
        env = Game2048Env()
        env.board = board.copy()
        env.score = score
        
        # 檢查每個可能的動作
        for action in range(4):
            if not env.is_move_legal(action):
                continue
                
            # 創建一個環境副本進行模擬
            sim_env = Game2048Env()
            sim_env.board = board.copy()
            sim_env.score = score
            
            # 執行動作
            next_board, next_score, done, _ = sim_env.step(action)
            
            # 如果動作有效
            if sim_env.last_move_valid:
                # 計算這個動作的期望值
                move_score = self.expectimax(next_board, next_score, self.depth, False)
                
                if move_score > best_score:
                    best_score = move_score
                    best_action = action
        
        # 如果沒有合法動作，隨機選擇
        if best_action == -1:
            legal_actions = []
            for action in range(4):
                if env.is_move_legal(action):
                    legal_actions.append(action)
            
            best_action = random.choice(range(4)) if not legal_actions else random.choice(legal_actions)
        
        # 快取結果
        self.move_cache[board_key] = best_action
        
        return best_action
    
    def expectimax(self, board, score, depth, is_max_player):
        """期望最大搜索算法"""
        # 達到搜索深度或遊戲結束
        if depth == 0:
            return Heuristics.evaluate(board, score)
        
        # 創建環境
        env = Game2048Env()
        env.board = board.copy()
        env.score = score
        
        if is_max_player:  # 玩家回合（選擇最佳動作）
            max_score = -float('inf')
            
            # 嘗試所有可能的動作
            for action in range(4):
                if not env.is_move_legal(action):
                    continue
                
                # 創建一個環境副本進行模擬
                sim_env = Game2048Env()
                sim_env.board = board.copy()
                sim_env.score = score
                
                # 執行動作
                next_board, next_score, done, _ = sim_env.step(action)
                
                if sim_env.last_move_valid:
                    # 遞迴評估
                    move_score = self.expectimax(next_board, next_score, depth - 1, False)
                    max_score = max(max_score, move_score)
            
            # 如果沒有合法動作
            if max_score == -float('inf'):
                return Heuristics.evaluate(board, score)
                
            return max_score
            
        else:  # 電腦回合（隨機生成新磚塊）
            # 找到所有空格
            empty_cells = list(zip(*np.where(board == 0)))
            
            if not empty_cells:  # 沒有空格
                return Heuristics.evaluate(board, score)
            
            # 2和4出現的概率
            probabilities = [0.9, 0.1]  # 90% 是2，10% 是4
            tile_values = [2, 4]
            
            avg_score = 0
            
            # 由於空格太多可能計算量過大，我們取樣計算
            samples = min(len(empty_cells), 3)  # 限制樣本數
            sampled_cells = random.sample(empty_cells, samples)
            
            for i, j in sampled_cells:
                for val, prob in zip(tile_values, probabilities):
                    # 創建新板
                    new_board = board.copy()
                    new_board[i, j] = val
                    
                    # 遞迴評估
                    child_score = self.expectimax(new_board, score, depth - 1, True)
                    avg_score += prob * child_score / samples
            
            return avg_score

# 蒙特卡洛樹搜索節點
class MCTSNode:
    def __init__(self, board, score, parent=None, action=None):
        self.board = board.copy()
        self.score = score
        self.parent = parent
        self.action = action  # 從父節點到達此節點的動作
        self.children = {}  # action -> MCTSNode
        self.visits = 0
        self.value = 0
        self.untried_actions = []  # 尚未嘗試的動作
        
        # 初始化可能的動作
        env = Game2048Env()
        env.board = board.copy()
        env.score = score
        
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]
        
    def uct_select_child(self, exploration=1.414):
        """使用UCB1公式選擇子節點"""
        if not self.children:
            return None
            
        log_visits = math.log(self.visits)
        
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            # UCB1公式
            exploit = child.value / child.visits if child.visits > 0 else 0
            explore = exploration * math.sqrt(log_visits / child.visits) if child.visits > 0 else float('inf')
            score = exploit + explore
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_child, best_action
    
    def expand(self):
        """從未嘗試的動作中擴展一個節點"""
        if not self.untried_actions:
            return None
            
        action = self.untried_actions.pop()
        
        # 創建環境並執行動作
        env = Game2048Env()
        env.board = self.board.copy()
        env.score = self.score
        
        next_board, next_score, done, _ = env.step(action)
        
        # 創建子節點
        child = MCTSNode(next_board, next_score, self, action)
        self.children[action] = child
        
        return child
    
    def update(self, result):
        """更新節點統計信息"""
        self.visits += 1
        self.value += result

# 蒙特卡洛樹搜索代理
class MCTSAgent:
    def __init__(self, iterations=1000, exploration=1.414):
        self.iterations = iterations
        self.exploration = exploration
    
    def get_action(self, board, score):
        """使用MCTS選擇最佳動作"""
        start_time = time.time()
        
        # 創建根節點
        root = MCTSNode(board, score)
        
        # 運行MCTS
        for _ in range(self.iterations):
            # 選擇
            node = self.select(root)
            
            # 擴展
            if node.untried_actions:
                node = node.expand()
                
            # 模擬
            simulation_result = self.simulate(node.board, node.score)
            
            # 反向傳播
            self.backpropagate(node, simulation_result)
            
            # 檢查時間限制
            if time.time() - start_time > 0.9:  # 90% 的可用時間
                break
                
        # 選擇訪問次數最多的動作
        best_action = -1
        best_visits = -1
        
        for action, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        
        # 如果沒有找到動作，使用啟發式選擇
        if best_action == -1:
            env = Game2048Env()
            env.board = board.copy()
            env.score = score
            
            # 使用Expectimax啟發式
            expectimax = ExpectimaxAgent(depth=1)
            best_action = expectimax.get_action(board, score)
        
        return best_action
    
    def select(self, node):
        """選擇階段 - 使用UCT公式選擇節點直到找到可擴展的節點"""
        while node.untried_actions == [] and node.children:
            # 使用UCT選擇子節點
            node, _ = node.uct_select_child(self.exploration)
        return node
    
    def simulate(self, board, score, max_moves=100):
        """模擬階段 - 從給定狀態隨機播放直到遊戲結束"""
        env = Game2048Env()
        env.board = board.copy()
        env.score = score
        
        done = False
        moves = 0
        
        while not done and moves < max_moves:
            # 獲取合法動作
            legal_actions = [a for a in range(4) if env.is_move_legal(a)]
            
            if not legal_actions:
                break
                
            # 隨機選擇動作，但更偏好保持高值磚塊在角落的策略
            action_scores = []
            
            for action in legal_actions:
                test_env = Game2048Env()
                test_env.board = env.board.copy()
                test_env.score = env.score
                
                next_board, next_score, _, _ = test_env.step(action)
                action_score = Heuristics.evaluate(next_board, next_score)
                action_scores.append((action, action_score))
            
            # 根據評分概率選擇動作
            total_score = sum(max(1, score) for _, score in action_scores)
            probs = [max(1, score) / total_score for _, score in action_scores]
            
            selected_action = random.choices([a for a, _ in action_scores], weights=probs)[0]
            
            # 執行選中動作
            _, new_score, done, _ = env.step(selected_action)
            moves += 1
        
        # 返回得分增量作為結果
        return env.score - score
    
    def backpropagate(self, node, result):
        """反向傳播階段 - 將模擬結果傳回所有訪問過的節點"""
        while node is not None:
            node.update(result)
            node = node.parent

# 混合代理 - 結合Expectimax和MCTS
class HybridAgent:
    def __init__(self, expectimax_depth=3, mcts_iterations=500):
        self.expectimax = ExpectimaxAgent(depth=expectimax_depth)
        self.mcts = MCTSAgent(iterations=mcts_iterations)
        self.pattern_database = {}  # 存儲已知的模式和最佳動作
        
    def get_action(self, board, score):
        """使用混合策略選擇最佳動作"""
        
        # 1. 檢查模式數據庫
        board_key = board.tobytes()
        if board_key in self.pattern_database:
            return self.pattern_database[board_key]
        
        # 2. 獲取環境
        env = Game2048Env()
        env.board = board.copy()
        env.score = score
        
        # 3. 確定棋盤發展階段（基於最大磚塊和空格數）
        max_tile = np.max(board)
        empty_tiles = np.sum(board == 0)
        game_stage = "early"
        
        if max_tile >= 1024:
            game_stage = "late"
        elif max_tile >= 256 or empty_tiles <= 6:
            game_stage = "mid"
        
        # 4. 決定使用哪個代理
        if game_stage == "early":
            # 早期階段用啟發式快速選擇動作
            action = self.select_heuristic_action(board, score)
        elif game_stage == "mid":
            # 中期使用期望最大
            action = self.expectimax.get_action(board, score)
        else:
            # 後期使用MCTS進行深度搜索
            action = self.mcts.get_action(board, score)
        
        # 5. 儲存結果到模式數據庫
        self.pattern_database[board_key] = action
        
        return action
    
    def select_heuristic_action(self, board, score):
        """使用啟發式策略快速選擇動作"""
        env = Game2048Env()
        env.board = board.copy()
        env.score = score
        
        best_score = -float('inf')
        best_action = -1
        
        # 嘗試每個動作，選擇導致最佳評分的動作
        for action in range(4):
            if not env.is_move_legal(action):
                continue
                
            test_env = Game2048Env()
            test_env.board = board.copy()
            test_env.score = score
            
            next_board, next_score, _, _ = test_env.step(action)
            
            # 主要策略：保持大數在角落
            corner_weight = 4.0
            smoothness_weight = 1.0
            merge_weight = 1.0
            monotonicity_weight = 2.0
            empty_weight = 2.5
            
            # 計算各項指標
            corner_value = Heuristics.corner_max(next_board)
            smoothness = Heuristics.smoothness(next_board)
            merge_potential = Heuristics.merge_potential(next_board)
            monotonicity = Heuristics.monotonicity(next_board)
            empty_count = Heuristics.empty_tiles(next_board)
            
            # 加權組合
            action_score = (
                corner_weight * math.log2(corner_value + 1) +
                smoothness_weight * smoothness +
                merge_weight * merge_potential +
                monotonicity_weight * monotonicity +
                empty_weight * empty_count
            )
            
            if action_score > best_score:
                best_score = action_score
                best_action = action
        
        # 如果沒有找到好的動作，優先嘗試向上或向左
        if best_action == -1:
            preferred_actions = [0, 2, 3, 1]  # 上，左，右，下
            for action in preferred_actions:
                if env.is_move_legal(action):
                    return action
                    
            return random.randint(0, 3)  # 最後隨機選擇
            
        return best_action

# 主函數
def get_action(state, score):
    # 靜態變量保存代理實例
    if not hasattr(get_action, 'agent'):
        # 創建混合代理
        get_action.agent = HybridAgent(expectimax_depth=3, mcts_iterations=200)
    
    # 執行代理
    return get_action.agent.get_action(state, score)
def run_episode(env, render=False, render_interval=50):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_tile = 0
        
        while True:
            if render and (steps % render_interval == 0):
                env.render(action=None, savepath=f"frame_{steps:04d}.png")
            
            action = get_action(state, env.score)
            next_state, reward, done, _ = env.step(action)
            
            total_reward += reward
            steps += 1
            max_tile = max(max_tile, np.max(next_state))
            state = next_state

            # if steps % 10 == 0:
            #     print(state)
            #     print(reward)
            
            if done:
                break
                
        return env.score, max_tile, steps
if __name__ == "__main__":
    # main()
    import argparse
    import numpy as np
    from tqdm import tqdm
    from collections import defaultdict
    import subprocess

    url = "https://www.dropbox.com/scl/fi/3rldcscf4bmurav3z4tlg/2048_320k.bin?rlkey=q1k2fy44smjbpaos88egm3i2u&st=4fg86rp1&dl=0"
    output_file = "hello.bin"

    subprocess.run(["curl", "-L", "-o", output_file, url], check=True)
    from test import *
    random.seed(0)
    for i in range(10):
        env = Game2048Env()
        env.reset()
        print(run_episode(env))


