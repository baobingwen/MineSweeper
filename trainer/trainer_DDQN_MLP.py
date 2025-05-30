# trainer_DDQN_MLP.py
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from env_DQN import MineSweeperEnv

# --------------------------- DQN网络 ---------------------------
class DDQN(nn.Module):
    """
    DDQN,MLP模型
    
    采用三层全连接网络结构：
    输入层 -> 隐藏层 -> 输出层
    - 输入层大小为棋盘格子数量，输出层大小为动作数量（每个格子一个动作）。
    - 输入层：每个格子是一个特征（未揭开、地雷、数字等）
    - 输出层：每个格子对应一个动作（选择点击某格子）。
    """
    def __init__(self, input_size, output_size):
        super(DDQN, self).__init__()
        # 输入层：每个格子是一个特征
        self.fc1 = nn.Linear(input_size, 128)  # 输入大小根据棋盘尺寸而定，通常是9x9 = 81
        self.fc2 = nn.Linear(128, 64)  # 隐藏层
        self.fc3 = nn.Linear(64, output_size)  # 输出层：每个格子一个动作（选择点击某格子）

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# --------------------------- 经验回放 ---------------------------
class ReplayBuffer:
    """
    经验回放池，用来存储状态、动作、奖励和下一状态。
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# --------------------------- DQN Agent ---------------------------
class DQNAgent:
    """
    DQN智能体类，包含选择动作、更新Q值、训练等功能。
    """
    def __init__(self, 
                input_size:int, output_size:int, 
                gamma=0.95, 
                epsilon_start=1.0, epsilon_final=0.1, epsilon_decay=200):
        self.input_size = input_size
        self.output_size = output_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay

        # 初始化网络和优化器
        self.policy_net = DDQN(input_size, output_size).cuda()  # 使用GPU训练
        self.target_net = DDQN(input_size, output_size).cuda()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

        # 初始化经验回放池
        self.memory = ReplayBuffer(10000)

        # 复制目标网络参数
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def select_action(self, state, episode):
        """
        根据epsilon-greedy策略选择动作。
        """
        # 获取有效动作的掩码（未揭开的格子）
        valid_mask = (state == -2)
        
        if not np.any(valid_mask):
            # 如果没有有效动作，返回随机动作
            return random.choice(range(self.output_size))
        
        # 逐步减少 epsilon（探索与利用平衡）
        epsilon = self.epsilon_final + (self.epsilon - self.epsilon_final) * \
                  np.exp(-1. * episode / self.epsilon_decay)

        if random.random() > epsilon:
            # 利用策略：选择最大Q值对应的动作
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).cuda()
                q_values = self.policy_net(state_tensor)
                valid_mask_tensor = torch.tensor(valid_mask, dtype=torch.bool).cuda()
                masked_q = torch.where(
                    valid_mask_tensor,
                    q_values,
                    torch.tensor(-np.inf).cuda()  # 对无效动作的Q值设为负无穷
                )
                action = masked_q.argmax().item()
        else:
            # 随机选择动作（探索）
            valid_indices = np.flatnonzero(valid_mask)
            action = np.random.choice(valid_indices)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        """
        存储一条经验到经验回放池。
        """
        state = np.array(state).flatten()  # 将状态展平为一维数组
        next_state = np.array(next_state).flatten()  # 将下一状态展平为一维数组
        self.memory.push(state, action, reward, next_state, done)

    def optimize_model(self, batch_size:int =64):
        """
        进行模型优化：更新Q值。
        """
        if self.memory.size() < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        batch = list(zip(*transitions))
        state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32).cuda()
        action_batch = torch.tensor(np.array(batch[1]), dtype=torch.long).cuda()
        reward_batch = torch.tensor(np.array(batch[2]), dtype=torch.float32).cuda()
        next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float32).cuda()
        done_batch = torch.tensor(np.array(batch[4]), dtype=torch.bool).cuda()

        # 计算目标Q值，DDQN，使用双网络来计算目标Q值
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # 使用策略网络选择动作
            next_actions = self.policy_net(next_state_batch).argmax(1)
            # 使用目标网络评估Q值
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)

        # 计算损失
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0) # 梯度裁剪，防止梯度爆炸
        self.optimizer.step()

    def update_target_network(self):
        """
        更新目标网络的权重。
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path:str) -> None:
        """
        保存模型到本地。
        
        :param path: 模型保存路径
        """
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """
        加载模型。
        """
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()
        print(f"Model loaded from {path}")


# --------------------------- 训练部分 ---------------------------
def train_dqn(agent:DQNAgent, num_episodes:int =1000) -> None:
    """
    训练DQN智能体。
    """

    total_steps = 0
    base_path = './trainer/models/'
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        episode_steps = 0  # 记录当前episode的步数

        while not done:
            action = agent.select_action(state, episode)
            next_state, reward, done, _, _ = env.step(action)  # 根据动作与环境交互
            agent.store_transition(state, action, reward, next_state, done)
            agent.optimize_model()
            state = next_state
            total_reward += reward

            # 更新步数计数器
            total_steps += 1
            episode_steps += 1

            # 每隔50000步数保存模型
            if total_steps % 50000 == 0:
                formatted_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"{base_path}ddqn_model_MLP_step{total_steps}_{formatted_time}.pth"
                agent.save_model(path)
                print(f"已保存第 {total_steps} 步的模型到 {path}")

        # 每隔一定步数更新目标网络
        if episode % 5 == 0:
            agent.update_target_network()

        print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Episode Steps: {episode_steps}, Total Steps: {total_steps}")

# --------------------------- 测试部分 ---------------------------
def test_dqn(agent, num_episodes=10):
    """
    测试DQN智能体。
    """
    total_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state, episode)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Test Episode {episode}/{num_episodes}, Total Reward: {total_reward}")

    avg_reward:float = sum(total_rewards) / len(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")

# --------------------------- 主程序 ---------------------------
if __name__ == "__main__":
    # 假设扫雷环境尺寸是9x9
    input_size:int = 81  # 9x9格子 = 81
    output_size:int = 81  # 每个格子是一个动作

    # 初始化 DQN智能体
    agent:DQNAgent = DQNAgent(input_size, output_size)

    # 创建扫雷环境（使用env_DQN打包好的环境）
    env:MineSweeperEnv = MineSweeperEnv()

    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        print("CUDA is available. Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")

    # 测试模型
    #test_dqn(agent, num_episodes=10)
    #print('Testing completed.')

    # 训练 DQN 智能体
    num_episodes:int = 10_000_000  # 训练的总轮数
    train_dqn(agent, num_episodes=num_episodes)
    print(f"Training completed after {num_episodes} episodes.")

    # 保存模型
    formatted_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = "./trainer/models/"
    path = f"{base_path}ddqn_model_MLP_{formatted_time}_{num_episodes}ep.pth"
    agent.save_model(path)
    print("Model saved.")