# mine_logic.py
import random
from typing import List, Tuple

class MineLogic:
    def __init__(self, width: int = 9, height:int = 9, mine_num: int = 10) -> None:
        """
        初始化扫雷游戏核心逻辑模块
        
        :param width: 雷区宽度（列数）
        :param height: 雷区高度（行数）
        :param mine_num: 地雷总数
        """
        # 游戏参数
        self._width = width
        self._height = height
        self._mine_num = mine_num

        # 游戏状态常量
        self.STATE_NEW_GAME = 'new_game'
        self.STATE_RUNNING = 'running'
        self.STATE_LOSE = 'lose'
        self.STATE_WIN = 'win'

        # 初始化游戏状态
        self.game_state = self.STATE_NEW_GAME
        self.start_time = 0.0 # 游戏开始时间
        self.elapsed_time = 0.0 # 游戏经过时间
        
        # 雷格状态
        self.mine_board: List[List[str]] = []     # 真实雷区（'*'=雷，数字=周围雷数，' '空格=空白）
        self.cover_board: List[List[str]] = []    # 覆盖状态（' '=未揭开，'1'=旗标，'*'=已揭开）
        self.ghost_board: List[List[bool]] = []    # 虚化状态（用于临时显示），True=虚化，False=正常显示
        
        # 游戏计数器
        self.mine_remain_num = self._mine_num # 剩余雷数
        
        # 所有随单局改变的量，都要存在change里
        self.reset_board(width,height,mine_num) 

    def reset_board(self,width=9,height=9,mine_num=10) -> None:
        """重置游戏到初始状态"""
        self.game_state = 'new_game'
        self.clicking = False # 左击有效格子
        self._mine_num = mine_num
        self.mine_remain_num = self._mine_num
        
        self.start_time = 0.0
        self.now_time = '0'
        
        self.mine_board = [[' ' for _ in range(width)] for _ in range(height)] # ' '空白,'数字'数字,'*'雷
        self.mine_cover_board = [[' ' for _ in range(width)] for _ in range(height)] # ' '空白未扫,'1'标记,'*'已扫
        self.ghost_board = [[False for _ in range(width)] for _ in range(height)] # True虚化
        
        # 自动扫雷状态跟踪属性
        self.last_board_hash = None  # 上次局面哈希值
        self.need_auto_sweep = False  # 是否需要触发自动扫雷
        
        # 重置标记
        self.auto_sweep_used = self.auto_sweep_state  # 自动扫雷标记
        self.semi_auto_used = self.semi_auto_mode   # 辅助扫雷标记
        
        # 鼠标点击次数
        self.click_cnt = 0
        self.left_click_cnt = 0
        self.right_click_cnt = 0

    def generate_mines(self,click_x,click_y):
        # 防止溢出
        if self._mine_num > self._width * self._height - 9:
            return
        
        # 随机放置雷
        mines_placed_num = 0
        while mines_placed_num < self._mine_num:
            x = random.randint(0, self._height - 1)
            y = random.randint(0, self._width - 1)
            # 如果该位置已经有雷，则重新选择
            if self.mine_board[x][y] == '*':
                continue
            # 如果在鼠标点击格子3*3区域内，重新选择
            if click_x-1<=x<=click_x+1 and click_y-1<=y<=click_y+1:
                continue
            self.mine_board[x][y] = '*'  # -1表示有雷
            mines_placed_num += 1

        # 雷周围的数字
        for i in range(self._height):
            for j in range(self._width):
                if self.mine_board[i][j] == '*':
                    continue
                # 检查周围8个格子
                tmp=0
                for x in range(max(0, i - 1), min(self._height, i + 2)):
                    for y in range(max(0, j - 1), min(self._width, j + 2)):
                        if self.mine_board[x][y] == '*':
                            tmp+=1
                if tmp > 0:
                    self.mine_board[i][j] = str(tmp)

    def clear_blank(self,x,y):
        visited_area = [[0 for _ in range(self._width)] for _ in range(self._height)]
        def clear(i,j):
            if visited_area[i][j] == 1:
                return
            visited_area[i][j] = 1
            
            # 如果本格真实层是数字,只清除本格覆盖层后返回
            if self.mine_board[i][j].isdigit(): 
                self.mine_cover_board[i][j] = '*'
                return 
            
            # 本格真实层是空白，清除含本格内3*3区域格子
            clear(min(i + 1, self._height - 1), j)
            clear(min(i + 1, self._height - 1), min(j + 1, self._width - 1))
            clear(min(i + 1, self._height - 1), max(j - 1, 0))
            clear(max(i - 1, 0),j)
            clear(max(i - 1, 0), min(j + 1, self._width - 1))
            clear(max(i - 1, 0), max(j - 1, 0))
            clear(i, min(j + 1, self._width - 1))
            clear(i, max(j - 1, 0))
            
            self.mine_cover_board[i][j] = '*'
        clear(x,y)
    
    def game_over(self,x=-1,y=-1):
        
        if x==-1 and y==-1:
            for i in range(self._height):
                for j in range(self._width):
                    if self.mine_board[i][j] == '*' and self.mine_cover_board[i][j] != '1':
                        self.game_over_cell_x = i
                        self.game_over_cell_y = j
                        break
        else:
            self.game_over_cell_x=x
            self.game_over_cell_y=y
        
        self.running_state = False
        self.game_state = 'lose'
        self.mine_remain_num = self._mine_num
        
        # 显示所有雷
        for i in range(self._height):
            for j in range(self._width):
                if self.mine_board[i][j] == '*' and self.mine_cover_board[i][j] != '1':
                    self.mine_cover_board[i][j] = '*'
                elif self.mine_board[i][j] != '*' and self.mine_cover_board[i][j] == '1':
                    self.mine_cover_board[i][j] = '*'
        
        self.draw()
    
    def game_win(self):
        self.running_state = False
        self.mine_remain_num = self._mine_num
        self.game_state = 'win'
        
        # 去除所有真实层不为雷覆盖层空白未扫，将所有真实层为雷的覆盖层标记
        for i in range(self._height):
            for j in range(self._width):
                if self.mine_board[i][j] != '*' and self.mine_cover_board[i][j] == ' ':
                    self.mine_cover_board[i][j] = '*'
                if self.mine_board[i][j] == '*':
                    self.mine_cover_board[i][j] = '1'
        
        self.draw()
        
        # 更新胜利记录
        from records import records
        try:
            elapsed_time = float(self.now_time)
            records.add_record(self.difficulty, 
                                self.auto_sweep_used, 
                                self.semi_auto_used, 
                                elapsed_time
                            )
        except ValueError:
            print('ValueError')
            pass