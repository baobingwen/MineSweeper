# play_minesweeper.py
import random
import numpy as np
import torch
import torch.nn as nn
from env_DQN import MineSweeperEnv

# 加载DQN模型和代理
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.policy_net = DQN(input_size, output_size)
    
    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()
        print(f"Model loaded from {path}")
    
    def select_action(self, state):
        # 创建有效动作掩码（-2 表示未揭开的格子）
        valid_mask = (state == -2)
        
        # 如果没有有效动作，随机选择
        if not np.any(valid_mask):
            return random.randint(0, self.output_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = self.policy_net(state_tensor)
            
            # 将无效动作的Q值设为负无穷大
            q_values[~valid_mask] = -float('inf')
            
            # 选择最高Q值的有效动作
            return q_values.argmax().item()

def visualize_board(board, rows=9, cols=9):
    """可视化扫雷棋盘"""
    symbols = {
        -2: "■",  # 未打开
        -1: "💣", # 地雷
        0: " ",   # 空白
        1: "1️⃣", 2: "2️⃣", 3: "3️⃣", 
        4: "4️⃣", 5: "5️⃣", 6: "6️⃣", 
        7: "7️⃣", 8: "8️⃣"
    }
    
    print("+" + "---+" * cols)
    for i in range(rows):
        row_str = "|"
        for j in range(cols):
            cell = board[i * cols + j]
            row_str += f" {symbols[cell]} |"
        print(row_str)
        print("+" + "---+" * cols)

def play_game(agent, rows=9, cols=9):
    """使用训练好的模型玩扫雷游戏"""
    env = MineSweeperEnv(rows, cols, 10)
    state, _ = env.reset()
    done = False
    step = 0
    
    print("===== 扫雷游戏开始! =====")
    
    while not done:
        step += 1
        action = agent.select_action(state)
        row, col = divmod(action, cols)
        
        print(f"\n步骤 {step}: 点击位置 ({row}, {col})")
        
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        
        # 可视化当前棋盘
        visualize_board(state, rows, cols)
        
        # 显示奖励
        if done:
            game_state = env.game.get_game_state()
            if game_state == 1:
                print("🎉 恭喜你赢了! 🎉")
            else:
                print("💥 踩到地雷了! 游戏结束。")
            print(f"最终奖励: {reward}")
        else:
            print(f"本次奖励: {reward}")

def test_game(agent, rows=9, cols=9, mines=10):
    """
    测试一局扫雷游戏，使用训练好的DQN模型进行决策并可视化棋盘状态。
    """
    env = MineSweeperEnv(rows, cols, mines)
    state, _ = env.reset()
    done = False
    step = 0
    
    print("=" * 50)
    print(f"开始扫雷游戏 ({rows}x{cols}, {mines}个地雷)")
    print("=" * 50)
    
    while not done:
        step += 1
        action = agent.select_action(state)
        row, col = divmod(action, cols)
        
        # 检查动作是否有效
        if state[action] != -2:
            print(f"⚠️ 警告：AI选择了无效动作 ({row}, {col})！该位置已揭开")
        
        print(f"\n步骤 {step}: 点击位置 ({row}, {col})")
        
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        
        # 可视化当前棋盘
        visualize_board(state, rows, cols)
        
        # 显示奖励
        print(f"本次奖励: {reward:.1f}")
        
        if done:
            game_state = env.game.get_game_state()
            if game_state == 1:
                print("\n🎉 游戏胜利! 恭喜AI获胜! 🎉")
            else:
                print("\n💥 游戏结束! AI踩到地雷了! 💥")
            
            # 显示隐藏的地雷位置
            hidden_board = env.game.get_hidden_board()
            flat_board = np.array(hidden_board).flatten()
            print("\n完整棋盘（显示地雷位置）:")
            visualize_board(flat_board, rows, cols)
            
            print(f"最终奖励: {reward:.1f}")
            print(f"总步数: {step}")
    
    return step, reward

if __name__ == "__main__":
    # 设置棋盘大小
    rows, cols = 9, 9
    input_size = rows * cols
    output_size = rows * cols

    # 创建代理并加载模型
    agent = DQNAgent(input_size, output_size)
    base_dir = "./trainer/models/"
    model_name = "ddqn_model_MLP_step450000_20250530_183157.pth" # 每次测试时替换为模型文件名
    path = f"{base_dir}{model_name}"
    agent.load_model(path)  # 加载训练好的模型

    # 测试游戏
    test_game(agent, rows, cols, mines=10)

    # 开始游戏
    # play_game(agent, rows, cols)