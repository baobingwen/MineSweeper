# env_DQN.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# 假设你的 Rust 游戏逻辑已经通过 Python 绑定提供了一个类 MineSweeperCore
from core import MineSweeperCore

class MineSweeperEnv(gym.Env):
    def __init__(self, rows=9, cols=9, mines=10):
        super(MineSweeperEnv, self).__init__()
        self.rows = rows
        self.cols = cols
        self.mines = mines

        # 初始化游戏核心逻辑
        self.game = MineSweeperCore(rows, cols, mines)

        # 定义动作空间和观察空间
        self.action_space = spaces.Discrete(rows * cols)
        self.observation_space = spaces.Box(low=-2, high=8, shape=(rows * cols,), dtype=np.int32)

    def reset(self) -> tuple[np.ndarray, dict]:
        '''
        重置游戏
        '''

        self.game.reset()
        # 获取初始状态并转换为一维数组
        state: np.ndarray = np.array(self.game.get_visible_board()).flatten()
        return state, {}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        # 将动作从一维索引转换为二维坐标
        row, col = divmod(action, self.cols)
        
        # 检查动作是否有效（是否点击已揭开的格子）
        current_state = np.array(self.game.get_visible_board()).flatten()
        if current_state[action] != -2:  # -2 表示未揭开
            # 无效动作惩罚
            return current_state, -0.1, False, False, {}
        
        # 执行动作
        self.game.open_cell(row, col)
        # 获取新的状态、奖励和游戏结束标志
        state = np.array(self.game.get_visible_board()).flatten()
        reward = self.game.calculate_reward()
        done = self.game.get_game_state() != 0
        return state, reward, done, False, {}  # 返回额外信息，如调试信息，设置为 False
