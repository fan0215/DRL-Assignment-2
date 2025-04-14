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

# 主函數，取代原始的get_action函數
def get_action(state, score):
    # 如果模型文件存在，嘗試加載預訓練模型
    model_file = '2048_dqn_model.pth'
    if not hasattr(get_action, 'agent'):
        get_action.agent = DQNAgent(load_model=os.path.exists(model_file))
        get_action.env = Game2048Env()
        get_action.env.board = state.copy()
        get_action.env.score = score
    else:
        get_action.env.board = state.copy()
        get_action.env.score = score
    
    # 預處理狀態
    state_processed = preprocess_state(state, score)
    
    # 選擇動作
    action = get_action.agent.select_action(state_processed, get_action.env)
    
    return action

# 如果單獨運行此文件，則訓練代理
if __name__ == "__main__":
    agent, scores = train_agent(episodes=1000)
    plt.plot(scores)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig('training_progress.png')
    plt.show()


