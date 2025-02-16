# -*- coding: utf-8 -*-

import random
import sys
import pygame

from config import Config
from assets import assets

# 难度类别按钮
class DifficultyCategory():
    def __init__(self,center,surface,difficulty = '基础'):
        self.difficulty = difficulty
        self.surface = surface
        self.center = center
        
        self._load_difficulty_params()
        
        self.font = assets.fonts["button_difficulty"]
        self.txt_surface = self.font.render(
            self.difficulty,
            True,
            Config.COLORS['black']
        )
        self.rect = self.txt_surface.get_rect(center=center)
        #print(self.rect)
    
    def _load_difficulty_params(self):
        """从 Config 加载难度参数，彻底解耦配置"""
        # 获取预设难度配置
        params = Config.DIFFICULTIES.get(self.difficulty, {})

        # 设置参数（兼容自定义模式）
        self.wide_cell_num = params.get("width", None)
        self.high_cell_num = params.get("height", None)
        self.mine_num = params.get("mines", None)
    
    def handle_event(self,event,mine_area):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            if self.difficulty in Config.DIFFICULTIES:
                mine_area.change_difficulty(self.wide_cell_num,self.high_cell_num,self.mine_num,self.difficulty)
            elif self.difficulty == '自定义': # 自定义
                mine_area.change_difficulty(difficulty = self.difficulty)
    
    def draw(self):
        self.surface.blit(self.txt_surface,self.rect)

# 自定义模式总类
class Custom():
    def __init__(self,button_width,button_height,button_mine_num,button_confirm):
        self.button_width = button_width
        self.button_height = button_height
        self.button_mine_num = button_mine_num
        self.button_confirm = button_confirm
    
    def handle_event(self,mine_area):
        if self.button_confirm.clicked == False:
            return
        
        if mine_area.difficulty != '自定义':
            return
        
        mine_area.change(self.button_width.num,self.button_height.num,self.button_mine_num.num)

# 自定义模式用于显示和改变雷的数量、宽、高的输入框
class CustomButton():
    def __init__(self,name,surface,init_num,left):
        self.surface = surface
        self.name = name
        
        self.num_font = assets.fonts["num_custom_button"]
        self.name_font = assets.fonts["name_custom_button"]
        
        self.num = init_num
        self.left = left
        self.top = Config.SIZE["CUSTOM_BUTTON_TOP"]
        
        self.change_element()
        self.change_layout()
    
    def change_element(self):
        self.up_clicked = False
        self.down_clicked = False
    
    def change_layout(self):
        self.width = Config.SIZE["CUSTOM_BUTTON_WIDTH"]
        self.height = Config.SIZE["CUSTOM_BUTTON_HEIGHT"]
        self.rect = pygame.rect.Rect(self.left,self.top,self.width,self.height)
        
        self.num_width = len(str(self.num)) * 22
        self.num_left = self.left + self.width // 2 - self.num_width // 2
        self.num_top = self.top + 8
        self.num_height = self.height - 4
        self.num_rect = pygame.rect.Rect(self.num_left,self.num_top,self.num_width,self.num_height)
        
        # 调节按钮
        self.tri_gap_half = 3
        # 上三角调节按钮
        self.tri_up_width = 12
        self.tri_up_height = 10
        self.tri_up_left = self.left + self.width * 4 // 5
        self.tri_up_top = self.top + self.height // 2 - self.tri_gap_half - self.tri_up_height
        self.tri_up_rect = pygame.rect.Rect(self.tri_up_left,self.tri_up_top,self.tri_up_width,self.tri_up_height)
        
        self.tri_up_point_sequence = ((self.tri_up_left + self.tri_up_width // 2,self.tri_up_top),
                                    (self.tri_up_left,self.tri_up_top + self.tri_up_height),
                                    (self.tri_up_left + self.tri_up_width, self.tri_up_top + self.tri_up_height))
        # 下三角调节按钮
        self.tri_down_width = self.tri_up_width
        self.tri_down_height = self.tri_up_height
        self.tri_down_left = self.tri_up_left
        self.tri_down_top = self.top + self.height // 2 + self.tri_gap_half
        self.tri_down_rect = pygame.rect.Rect(self.tri_down_left,self.tri_down_top,self.tri_down_width,self.tri_down_height)
        
        self.tri_down_point_sequence = ((self.tri_down_left,self.tri_down_top),
                                        (self.tri_down_left + self.tri_down_width,self.tri_down_top),
                                        (self.tri_down_left + self.tri_down_width // 2, self.tri_down_top + self.tri_down_height))
        
        self.name_width = len(self.name) * 22
        self.name_height = self.height
        self.name_left = self.left - self.name_width
        self.name_top = self.top
        self.name_rect = pygame.rect.Rect(self.name_left,self.name_top,self.name_width,self.name_height)
    
    def draw(self,mine_area):
        if mine_area.difficulty != '自定义':
            return
        # 名
        txt_surface = self.name_font.render(self.name,True,Config.COLORS['black'])
        txt_rect = txt_surface.get_rect(center=self.name_rect.center)
        self.surface.blit(txt_surface,txt_rect)
        
        # 输入框
        pygame.draw.rect(self.surface,Config.COLORS['background'],self.rect)
        pygame.draw.rect(self.surface,Config.COLORS['black'],self.rect,2,-1,3,3,3,3)
        
        # 数字显示
        txt_surface = self.num_font.render(str(self.num),True,Config.COLORS['black'])
        txt_rect = txt_surface.get_rect(center=self.num_rect.center)
        self.surface.blit(txt_surface,txt_rect)
        
        # 三角形调节按钮
        # 上
        if self.up_clicked == True or self.tri_up_rect.collidepoint(pygame.mouse.get_pos()): # 点击或悬浮
            pygame.draw.polygon(self.surface,Config.COLORS['tri_button_dark'],self.tri_up_point_sequence)
        else: # 未点击，未悬浮
            pygame.draw.polygon(self.surface,Config.COLORS['tri_button_shadow'],self.tri_up_point_sequence)
        # 下
        if self.down_clicked == True or self.tri_down_rect.collidepoint(pygame.mouse.get_pos()):
            pygame.draw.polygon(self.surface,Config.COLORS['tri_button_dark'],self.tri_down_point_sequence)
        else:
            pygame.draw.polygon(self.surface,Config.COLORS['tri_button_shadow'],self.tri_down_point_sequence)
    
    def handle_event(self,event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.tri_up_rect.collidepoint(event.pos): # 上
                self.up_clicked = True
                self.num += 1
            if self.tri_down_rect.collidepoint(event.pos):
                self.down_clicked = True
                self.num -= 1
        elif event.type == pygame.MOUSEBUTTONUP:
            self.up_clicked = False
            self.down_clicked = False

# 确定按钮
class ConfirmButton():
    def __init__(self,surface,left):
        self.surface = surface
        self.name = '确定'
        
        self.font = assets.fonts["num_custom_button"]
        self.name_font = assets.fonts["name_custom_button"]
        
        self.left = left
        self.top = Config.SIZE["CUSTOM_BUTTON_TOP"]
        
        self.change_element()
        self.change_layout()
    
    def change_element(self):
        self.up_clicked = False
        self.down_clicked = False
    
    def change_layout(self):
        self.width = 96
        self.height = 36
        self.rect = pygame.rect.Rect(self.left,self.top,self.width,self.height)
        
        self.name_width = len(self.name) * 22
        self.name_height = self.height
        self.name_left = self.left + self.width // 2 - self.name_width // 2
        self.name_top = self.top
        self.name_rect = pygame.rect.Rect(self.name_left,self.name_top,self.name_width,self.name_height)
    
    def draw(self,mine_area):
        if mine_area.difficulty != '自定义':
            return
        # 输入框
        pygame.draw.rect(self.surface,Config.COLORS['background'],self.rect)
        pygame.draw.rect(self.surface,Config.COLORS['black'],self.rect,2,-1,3,3,3,3)
        
        # 名
        txt_surface = self.name_font.render(self.name,True,Config.COLORS['black'])
        txt_rect = txt_surface.get_rect(center=self.name_rect.center)
        self.surface.blit(txt_surface,txt_rect)
    
    def handle_event(self,event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.clicked = True
        else:
            self.clicked = False

# restart 按钮
class Restart():
    def __init__(self,surface,rect):
        # 绘制量
        self.surface = surface
        self.side_length = 25
        self.rect = rect
        
        # 图标预加载
        self.image_normal = assets.images['restart_normal']
        self.image_normal_down = assets.images['restart_normal_down']
        self.image_win = assets.images['restart_win']
        self.image_win_down = assets.images['restart_win_down']
        self.image_lose = assets.images['restart_lose']
        self.image_lose_down = assets.images['restart_lose_down']
        #self.image_restart = pygame.image.load('./image/OIP.jpg')
        self.image_clicking = assets.images['restart_clicking']
        
        self.image = self.image_normal
        
        # 点击量
        self.mousebuttondown = False
    
    def draw(self,rect):
        self.surface.blit(self.image,rect)
        #self.surface.blit(self.txt_surface,rect)
    
    def handle_event(self,event,mine_area):
        self.rect = mine_area.restart_rect
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.mousebuttondown = True
                if mine_area.game_state == 'running' or mine_area.game_state == 'new_game':
                    self.image = self.image_normal_down
                elif mine_area.game_state == 'win': # 胜利点击按钮凹陷
                    self.image = self.image_win_down
                elif mine_area.game_state == 'lose':
                    self.image = self.image_lose_down
        if event.type == pygame.MOUSEBUTTONUP:
            if self.mousebuttondown == True:
                self.image = self.image_normal
                self.mousebuttondown = False
                mine_area.restart()
                return
    
    def handle_game_state(self,mine_area):
        if self.mousebuttondown == True: # 正在按下restart按钮，不做扫雷的反应
            return
        if mine_area.game_state == 'new_game': # 新游戏，尚未开始
            self.image = self.image_normal
        elif mine_area.game_state == 'running':
            if mine_area.clicking == True: # 点击数字或空白，restart图标做出惊讶反应
                self.image = self.image_clicking
            else:
                self.image = self.image_normal
        elif mine_area.game_state == 'win': # 胜利图标
            self.image = self.image_win
        elif mine_area.game_state == 'lose': # 失败图标
            self.image = self.image_lose

# 扫雷区域
class MineArea():
    def __init__(self, surface, width=9, height=9, mine_num=10):
        self.tot_width = width
        self.tot_height = height
        self.mine_num = mine_num
        
        self.difficulty = '基础'
        
        self.auto_sweep_state = False
        self.semi_auto_mode = True  # 手动扫雷 半自动模式开关
        
        # 图片资源
        self.image_hidden_1 = assets.images['hidden_1']
        self.image_hidden_2 = assets.images['hidden_2']
        self.image_hidden_3 = assets.images['hidden_3']
        self.image_hidden_4 = assets.images['hidden_4']
        self.image_hidden_5 = assets.images['hidden_5']
        self.image_hidden_6 = assets.images['hidden_6']
        self.image_hidden_7 = assets.images['hidden_7']
        self.image_hidden_8 = assets.images['hidden_8']
        self.image_hidden_mine_normal = assets.images['hidden_mine_normal']
        self.image_hidden_mine_triggar = assets.images['hidden_mine_triggar']
        
        self.image_cover_blank = assets.images['cover_blank']
        self.image_cover_flag = assets.images['cover_flag']
        
        # 显示
        self.num_font = pygame.font.Font(None,25)
        
        self.mine_num_font = assets.fonts["num_remaining_flags"]
        self.time_cnt_font = assets.fonts["num_timer"]
        
        self.surface = surface
        
        self.change_layout(width,height,mine_num)
        
        # 算法
        self.change_board(width,height,mine_num)
        
    
    def change_layout(self, width=9, height=9, mine_num=10):
        self.tot_width = width
        self.tot_height = height
        
        self.cell_side_length = Config.SIZE["CELL_SIDE_LENGTH"]
        self.cell_gap = Config.SIZE["CELL_GAP"]
        self.border = Config.SIZE["CELL_BORDER"]
        
        self.tot_display_width = self.tot_width * (self.cell_side_length + self.cell_gap) - self.cell_gap + self.border * 2
        self.left = 984 // 2 - self.tot_display_width // 2
        
        if self.difficulty == '自定义':
            self.top = 144
        else:
            self.top = 104
        
        # display_pad
        self.display_pad_left = self.left + self.border
        self.display_pad_top = self.top + self.border
        self.display_pad_width = (self.cell_side_length + self.cell_gap) * self.tot_width - self.cell_gap
        self.display_pad_height = 34
        self.display_pad_line_width = 2
        
        self.pad_count_gap_height = 5
        self.pad_count_gap_width = 1
        
        # count_area
        self.count_area_left = self.display_pad_left + self.display_pad_line_width + self.pad_count_gap_width
        self.count_area_top = self.display_pad_top + self.display_pad_line_width + self.pad_count_gap_height
        self.count_area_width = 39
        self.count_area_height = 23
        
        self.pad_restart_gap_height = self.pad_count_gap_height
        
        # restart button
        self.restart_width = self.cell_side_length
        self.restart_height = self.cell_side_length
        self.restart_left = self.display_pad_left + self.display_pad_width // 2 - self.restart_width // 2
        self.restart_top = self.display_pad_top + self.pad_restart_gap_height
        self.restart_rect = pygame.rect.Rect(self.restart_left,self.restart_top,self.restart_width,self.restart_height)
        
        # time cnt area
        self.time_cnt_width = self.count_area_width
        self.time_cnt_height = self.count_area_height
        self.time_cnt_left = self.display_pad_left + self.display_pad_width - self.pad_count_gap_width - self.count_area_width
        self.time_cnt_top = self.display_pad_top + self.display_pad_line_width + self.pad_count_gap_height
        
        self.display_pad_rect = pygame.rect.Rect(self.display_pad_left, self.display_pad_top, self.display_pad_width, self.display_pad_height)
        self.mine_num_rect = pygame.rect.Rect(self.count_area_left, self.count_area_top, self.count_area_width, self.count_area_height)
        self.time_cnt_rect = pygame.rect.Rect(self.time_cnt_left, self.time_cnt_top, self.time_cnt_width, self.time_cnt_height)
        
        self.pad_sweep_gap_height = 6
        
        # mine sweep area
        self.mine_sweep_line_width = 2
        self.mine_sweep_top = self.display_pad_top + self.display_pad_height + self.pad_sweep_gap_height
        self.mine_sweep_left = self.display_pad_left
        self.mine_sweep_width = (self.cell_side_length + self.cell_gap) * self.tot_width - self.cell_gap
        self.mine_sweep_height = (self.cell_side_length + self.cell_gap) * self.tot_height - self.cell_gap
        
        self.mine_sweep_rect = pygame.rect.Rect(self.mine_sweep_left,self.mine_sweep_top,self.mine_sweep_width,self.mine_sweep_height)
        
        # tot area
        self.tot_display_height = self.tot_height * (self.cell_side_length + self.cell_gap) - self.cell_gap + self.border * 2 + self.display_pad_height + self.pad_sweep_gap_height
        self.minearea_rect = pygame.rect.Rect(self.left,self.top, self.tot_display_width, self.tot_display_height)
    
    def change_board(self,width=9,height=9,mine_num=10):
        
        self.game_state = 'new_game'
        self.clicking = False # 左击有效格子
        self.mine_num = mine_num
        self.mine_remain_num = self.mine_num
        
        self.start_tick = pygame.time.get_ticks()
        self.now_time = '0'
        
        self.mine_board = [[' ' for _ in range(width)] for _ in range(height)] # ' '空白,'数字'数字,'*'雷
        self.mine_cover_board = [[' ' for _ in range(width)] for _ in range(height)] # ' '空白未扫,'1'标记,'*'已扫
        self.ghost_board = [[False for _ in range(width)] for _ in range(height)] # True虚化
        
        # 自动扫雷状态跟踪属性
        self.last_board_hash = None  # 上次局面哈希值
        self.need_auto_sweep = False  # 是否需要触发自动扫雷
    
    def change(self,width=9,height=9,mine_num=10):
        self.change_layout(width,height,mine_num)
        self.change_board(width,height,mine_num)
    
    def set_mine_location(self,click_x,click_y):
        # 防止溢出
        if self.mine_num > self.tot_width * self.tot_height - 9:
            return
        
        # 随机放置雷
        mines_placed_num = 0
        while mines_placed_num < self.mine_num:
            x = random.randint(0, self.tot_height - 1)
            y = random.randint(0, self.tot_width - 1)
            # 如果该位置已经有雷，则重新选择
            if self.mine_board[x][y] == '*':
                continue
            # 如果在鼠标点击格子3*3区域内，重新选择
            if click_x-1<=x<=click_x+1 and click_y-1<=y<=click_y+1:
                continue
            self.mine_board[x][y] = '*'  # -1表示有雷
            mines_placed_num += 1

        # 雷周围的数字
        for i in range(self.tot_height):
            for j in range(self.tot_width):
                if self.mine_board[i][j] == '*':
                    continue
                # 检查周围8个格子
                tmp=0
                for x in range(max(0, i - 1), min(self.tot_height, i + 2)):
                    for y in range(max(0, j - 1), min(self.tot_width, j + 2)):
                        if self.mine_board[x][y] == '*':
                            tmp+=1
                if tmp > 0:
                    self.mine_board[i][j] = str(tmp)
    
    def draw(self):
        # mine area
        pygame.draw.rect(self.surface,Config.COLORS['mine_area'],self.minearea_rect,width=0,border_radius=-1,border_top_left_radius=12,border_top_right_radius=10,border_bottom_left_radius=10,border_bottom_right_radius=10)
        
        # display pad
        #pygame.draw.rect(self.surface,MINEAREA_COLOR,self.display_pad_rect)
        pygame.draw.line(self.surface,Config.COLORS['gray'],self.display_pad_rect.topleft,self.display_pad_rect.topright,self.display_pad_line_width)
        pygame.draw.line(self.surface,Config.COLORS['gray'],self.display_pad_rect.topleft,self.display_pad_rect.bottomleft,self.display_pad_line_width)
        pygame.draw.line(self.surface,Config.COLORS['white'],self.display_pad_rect.topright,self.display_pad_rect.bottomright,self.display_pad_line_width)
        pygame.draw.line(self.surface,Config.COLORS['white'],self.display_pad_rect.bottomleft,self.display_pad_rect.bottomright,self.display_pad_line_width)
        
        # 剩余标记数显示区
        pygame.draw.rect(self.surface,Config.COLORS['black'],self.mine_num_rect)
        txt_surface = self.mine_num_font.render(str(self.mine_remain_num),True,Config.COLORS['red'])
        txt_rect = txt_surface.get_rect(center=self.mine_num_rect.center)
        self.surface.blit(txt_surface,txt_rect)
        
        # 计时显示区
        pygame.draw.rect(self.surface,Config.COLORS['black'],self.time_cnt_rect)
        txt_surface = self.time_cnt_font.render(self.now_time,True,Config.COLORS['red'])
        txt_rect = txt_surface.get_rect(center=self.time_cnt_rect.center)
        self.surface.blit(txt_surface,txt_rect)
        
        # mine sweep area
        pygame.draw.rect(self.surface,Config.COLORS['mine_area_dark'],self.mine_sweep_rect)
        pygame.draw.line(self.surface,Config.COLORS['gray'],(self.mine_sweep_rect.left-self.mine_sweep_line_width,self.mine_sweep_rect.top-self.mine_sweep_line_width),(self.mine_sweep_rect.left-self.mine_sweep_line_width,self.mine_sweep_rect.bottom+self.mine_sweep_line_width),self.mine_sweep_line_width)
        pygame.draw.line(self.surface,Config.COLORS['gray'],(self.mine_sweep_rect.left-self.mine_sweep_line_width,self.mine_sweep_rect.top-self.mine_sweep_line_width),(self.mine_sweep_rect.right+self.mine_sweep_line_width,self.mine_sweep_rect.top-self.mine_sweep_line_width),self.mine_sweep_line_width)
        pygame.draw.line(self.surface,Config.COLORS['white'],(self.mine_sweep_rect.left-self.mine_sweep_line_width,self.mine_sweep_rect.bottom+self.mine_sweep_line_width),(self.mine_sweep_rect.right+self.mine_sweep_line_width,self.mine_sweep_rect.bottom+self.mine_sweep_line_width),self.mine_sweep_line_width)
        pygame.draw.line(self.surface,Config.COLORS['white'],(self.mine_sweep_rect.right+self.mine_sweep_line_width,self.mine_sweep_rect.top-self.mine_sweep_line_width),(self.mine_sweep_rect.right+self.mine_sweep_line_width,self.mine_sweep_rect.bottom+self.mine_sweep_line_width),self.mine_sweep_line_width)
        
        # mine hidden layer
        for i in range(self.tot_height):
            for j in range(self.tot_width):
                # 此格位置属性
                self.mine_cell_left = self.mine_sweep_left + j * (self.cell_side_length + self.cell_gap)
                self.mine_cell_top = self.mine_sweep_top + i * (self.cell_side_length + self.cell_gap)
                self.mine_cell_width = self.cell_side_length
                self.mine_cell_height = self.cell_side_length
                self.mine_cell_rect = pygame.rect.Rect(self.mine_cell_left,self.mine_cell_top,self.mine_cell_width,self.mine_cell_height)
                # 绘制此格底板
                pygame.draw.rect(self.surface,Config.COLORS['mine_hidden_background'],self.mine_cell_rect)
                
                if self.mine_board[i][j] == '*': # 真实层为雷
                    # 绘制雷
                    image = self.image_hidden_mine_normal
                    self.surface.blit(image,self.mine_cell_rect)
                elif self.mine_board[i][j] == ' ': # 真实层为空白
                    pass
                else: # 真实层为数字
                    if self.mine_board[i][j] == '1':
                        image = self.image_hidden_1
                    elif self.mine_board[i][j] == '2':
                        image = self.image_hidden_2
                    elif self.mine_board[i][j] == '3':
                        image = self.image_hidden_3
                    elif self.mine_board[i][j] == '4':
                        image = self.image_hidden_4
                    elif self.mine_board[i][j] == '5':
                        image = self.image_hidden_5
                    elif self.mine_board[i][j] == '6':
                        image = self.image_hidden_6
                    elif self.mine_board[i][j] == '7':
                        image = self.image_hidden_7
                    elif self.mine_board[i][j] == '8':
                        image = self.image_hidden_8
                    self.surface.blit(image,self.mine_cell_rect)
        
        # mine cover layer
        for i in range(self.tot_height):
            for j in range(self.tot_width):
                if self.ghost_board[i][j] == True: # 覆盖层虚化，先绘制
                    # 此格位置属性
                    self.mine_cell_left = self.mine_sweep_left + j * (self.cell_side_length + self.cell_gap)
                    self.mine_cell_top = self.mine_sweep_top + i * (self.cell_side_length + self.cell_gap)
                    self.mine_cell_width = self.cell_side_length
                    self.mine_cell_height = self.cell_side_length
                    self.mine_cell_rect = pygame.rect.Rect(self.mine_cell_left,self.mine_cell_top,self.mine_cell_width,self.mine_cell_height)
                    # 绘制图案
                    pygame.draw.rect(self.surface,Config.COLORS['mine_hidden_background'],self.mine_cell_rect)
                    continue
                
                if self.mine_cover_board[i][j] == ' ': # 覆盖层是空白未扫
                    # 此格位置属性
                    self.mine_cell_left = self.mine_sweep_left + j * (self.cell_side_length + self.cell_gap)
                    self.mine_cell_top = self.mine_sweep_top + i * (self.cell_side_length + self.cell_gap)
                    self.mine_cell_width = self.cell_side_length
                    self.mine_cell_height = self.cell_side_length
                    self.mine_cell_rect = pygame.rect.Rect(self.mine_cell_left,self.mine_cell_top,self.mine_cell_width,self.mine_cell_height)
                    # 绘制图案
                    self.surface.blit(self.image_cover_blank,self.mine_cell_rect)
                elif self.mine_cover_board[i][j] == '1': # 覆盖层是标记
                    # 此格位置属性
                    self.mine_cell_left = self.mine_sweep_left + j * (self.cell_side_length + self.cell_gap)
                    self.mine_cell_top = self.mine_sweep_top + i * (self.cell_side_length + self.cell_gap)
                    self.mine_cell_width = self.cell_side_length
                    self.mine_cell_height = self.cell_side_length
                    self.mine_cell_rect = pygame.rect.Rect(self.mine_cell_left,self.mine_cell_top,self.mine_cell_width,self.mine_cell_height)
                    # 绘制图案
                    self.surface.blit(self.image_cover_flag,self.mine_cell_rect)
                
        
        # game over triggar mine
        if self.game_state == 'lose':
            # 此格位置属性
            self.mine_cell_left = self.mine_sweep_left + self.game_over_cell_y * (self.cell_side_length + self.cell_gap)
            self.mine_cell_top = self.mine_sweep_top + self.game_over_cell_x * (self.cell_side_length + self.cell_gap)
            self.mine_cell_width = self.cell_side_length
            self.mine_cell_height = self.cell_side_length
            self.mine_cell_rect = pygame.rect.Rect(self.mine_cell_left,self.mine_cell_top,self.mine_cell_width,self.mine_cell_height)
            
            self.surface.blit(self.image_hidden_mine_triggar,self.mine_cell_rect)
    
    def change_difficulty(self,width=9,height=9,mine_num=10,difficulty='基础'):
        self.difficulty = difficulty
        if difficulty == '自定义':
            self.change()
        else:
            self.change(width,height,mine_num)
    
    def restart(self):
        self.change(self.tot_width,self.tot_height,self.mine_num)
    
    def time_start(self):
        self.start_tick = pygame.time.get_ticks()

    def handle_time(self):
        if self.game_state == 'running': # 游戏已经开始
            self.now_time = str(round((pygame.time.get_ticks() - self.start_tick) / 1000,1)) # 现在显示时间
    
    # 自动扫雷
    def auto_sweep(self):
        if self.game_state != 'running': # 游戏不在运行时，不自动扫雷
            return
        
        if self.auto_sweep_state == False: # 未开启auto sweep，不自动扫雷
            return
        
        if self.need_auto_sweep == False: # auto_sweep运行中优化
            return
        
        # 生成当前局面哈希
        current_hash = hash(str(self.mine_cover_board))
        
        # 基本规则运行产生改变，尚不运行组合逻辑规则;基本规则运行无改变，执行组合逻辑规则
        if not self._run_basic_rules():
            self._run_combination_rules()
        
        # 更新局面哈希并重置触发标记
        new_hash = hash(str(self.mine_cover_board))
        if current_hash == new_hash:
            self.need_auto_sweep = False
        self.last_board_hash = new_hash
    
    # ===== 基本规则 =====
    def _run_basic_rules(self):
        
        # DEBUG
        #print(1,end='')
        
        basic_changed = False
        for i in range(self.tot_height):
            for j in range(self.tot_width):
                if self.mine_cover_board[i][j] != '*': # 覆盖层不是已扫
                    continue
                # 覆盖层为已扫
                if self.mine_board[i][j] == '*' or self.mine_board[i][j] == ' ': # 真实层为雷或空白
                    continue
                # 真实层为数字
                cnt_not_swept = 0
                cnt_signed = 0
                # 3*3区域内扫描
                for x in range(max(0, i - 1), min(self.tot_height, i + 2)):
                    for y in range(max(0, j - 1), min(self.tot_width, j + 2)):
                        if self.mine_cover_board[x][y] != '*':
                            cnt_not_swept += 1
                        if self.mine_cover_board[x][y] == '1':
                            cnt_signed += 1
                # 如果区域内不是已扫的格子数量等于真实层的数字，将这些格子覆盖层标记
                if cnt_not_swept == int(self.mine_board[i][j]):
                    for x in range(max(0, i - 1), min(self.tot_height, i + 2)):
                        for y in range(max(0, j - 1), min(self.tot_width, j + 2)):
                            if self.mine_cover_board[x][y] == ' ':
                                self.mine_cover_board[x][y] = '1'
                                self.mine_remain_num -= 1
                                basic_changed = True
                # 如果区域内标记格子数量等于真实层数字，将区域内空白未扫覆盖层的格子的覆盖层使用self.clear_blank()转化为已扫覆盖层
                if cnt_signed == int(self.mine_board[i][j]):
                    for x in range(max(0, i - 1), min(self.tot_height, i + 2)):
                        for y in range(max(0, j - 1), min(self.tot_width, j + 2)):
                            if self.mine_cover_board[x][y] == ' ':
                                self.clear_blank(x,y)
                                basic_changed = True
        # 基本规则运行成功，尚不运行组合逻辑规则
        return basic_changed
    
    # ===== 组合逻辑 =====
    def _run_combination_rules(self):
        
        # DEBUG
        #print(2,end='')
        
        # 遍历所有已扫的数字格子，寻找相邻格子对
        for i in range(self.tot_height):
            for j in range(self.tot_width):
                # 如果本格的覆盖层不是已扫，跳过
                if self.mine_cover_board[i][j] != '*': 
                    continue
                # 如果本格真实层不是数字，跳过
                if not self.mine_board[i][j].isdigit():
                    continue
                
                # 本格覆盖层已扫、真实层是数字
                # 检查相邻方向（右、下）
                directions = [(0,1), (1,0)]
                for dx, dy in directions:
                    ni, nj = i+dx, j+dy
                    # 如果不在范围内，跳过
                    if not (0 <= ni < self.tot_height and 0 <= nj < self.tot_width):
                        continue
                    # 如果邻格的覆盖层不是已扫 或 邻格真实层不是数字，跳过
                    if self.mine_cover_board[ni][nj] != '*' or not self.mine_board[ni][nj].isdigit():
                        continue
                    
                    # 邻格覆盖层已扫，真实层数字
                    # 获取两个格子的标记格子个数和未扫区域
                    a_unmarked = [(x,y) for x in range(i-1, i+2) for y in range(j-1, j+2)
                                if 0<=x<self.tot_height and 0<=y<self.tot_width
                                and self.mine_cover_board[x][y] == ' '] # 覆盖层空白未扫
                    
                    a_marked = sum(1 for x in range(i-1, i+2) for y in range(j-1, j+2)
                                if 0<=x<self.tot_height and 0<=y<self.tot_width
                                and self.mine_cover_board[x][y] == '1') # 覆盖层标记
                    
                    b_unmarked = [(x,y) for x in range(ni-1, ni+2) for y in range(nj-1, nj+2)
                                if 0<=x<self.tot_height and 0<=y<self.tot_width
                                and self.mine_cover_board[x][y] == ' ']
                    
                    b_marked = sum(1 for x in range(ni-1, ni+2) for y in range(nj-1, nj+2)
                                if 0<=x<self.tot_height and 0<=y<self.tot_width
                                and self.mine_cover_board[x][y] == '1')
                    
                    # 本格和邻格真实层数字与已标记格数之差，即此格剩余标记数
                    a_num = int(self.mine_board[i][j]) - a_marked
                    b_num = int(self.mine_board[ni][nj]) - b_marked
                    
                    # 包含关系
                    # 如果本格未扫区域包含邻格未扫区域 并且 本格剩余标记数等于邻格剩余标记数，清除未包含区域
                    if set(a_unmarked).issuperset(b_unmarked):
                        exclusive = list(set(a_unmarked) - set(b_unmarked))
                        if a_num == b_num:
                            for x,y in exclusive:
                                if self.mine_cover_board[x][y] == ' ':
                                    self.clear_blank(x, y)
                    if set(b_unmarked).issuperset(a_unmarked):
                        exclusive = list(set(b_unmarked) - set(a_unmarked))
                        if a_num == b_num:
                            for x,y in exclusive:
                                if self.mine_cover_board[x][y] == ' ':
                                    self.clear_blank(x, y)
                    
                    # 交叠关系
                    overlap = list(set(a_unmarked) & set(b_unmarked)) # 交叠格子坐标
                    a_excl = list(set(a_unmarked) - set(overlap)) # 本格独立区域雷数
                    b_excl = list(set(b_unmarked) - set(overlap)) # 邻格独立区域雷数
                    
                    diff = abs(a_num - b_num) # 本格区域剩余标记数与邻格区域剩余标记数之差
                    # 如果本格数字大于邻格数字 并且 本格未交叠区域格数等于两格数字之差
                    if a_num > b_num and len(a_excl) == diff:
                        for x,y in a_excl:
                            # 如果覆盖层为未扫 并且 剩下雷数大于零，标记本格覆盖层并将雷数-1
                            if self.mine_cover_board[x][y] == ' ' and self.mine_remain_num > 0:
                                self.mine_cover_board[x][y] = '1'
                                self.mine_remain_num -= 1
                    # 如果邻格数字大于本格数字 并且 邻格未交叠区域格数等于两格数字之差
                    elif b_num > a_num and len(b_excl) == diff:
                        for x,y in b_excl:
                            # 如果覆盖层为未扫 并且 剩下雷数大于零，标记本格覆盖层并将雷数-1
                            if self.mine_cover_board[x][y] == ' ' and self.mine_remain_num > 0:
                                self.mine_cover_board[x][y] = '1'
                                self.mine_remain_num -= 1
    
    def handle_event(self,event):
        if self.game_state == 'lose' or self.game_state == 'win':
            return
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # 左键
                
                # 超出扫雷区域，无反应
                if event.pos[0] > self.mine_sweep_rect.right or event.pos[0] < self.mine_sweep_rect.left or event.pos[1] < self.mine_sweep_rect.top or event.pos[1] > self.mine_sweep_rect.bottom:
                    return
                
                y = (event.pos[0] - self.mine_sweep_left) // (self.cell_side_length + self.cell_gap)
                x = (event.pos[1] - self.mine_sweep_top) // (self.cell_side_length + self.cell_gap)
                print(x,y,event.pos)
                
                # 左键被标记覆盖层，无反应
                if self.mine_cover_board[x][y] == '1':
                    return
                # 标记已扫
                #self.mine_cover_board[x][y] = '*'
                
                self.clicking = True
                
                if self.mine_cover_board[x][y] == ' ': # 左击覆盖层空白未扫
                    if self.game_state == 'new_game': # 游戏尚未开始，第一次左击空白
                        self.game_state = 'running' # 开始游戏
                        self.time_start()
                        self.set_mine_location(x,y) # 设置雷分布
                    if self.mine_board[x][y] == '*': # 左击雷，游戏失败
                        self.game_over(x,y)
                        return
                    elif self.mine_board[x][y] == ' ': # 左击空白，显示相连片的空白和边缘的数字
                        self.clear_blank(x,y)
                    else: # 左击数字，去除本格覆盖层
                        self.mine_cover_board[x][y] = '*'
                elif self.mine_cover_board[x][y] == '1': # 左击标记，无反应
                    pass
                elif self.mine_cover_board[x][y] == '*': # 左击已扫覆盖层
                    if self.mine_board[x][y] == '*': # 左击雷，无反应
                        pass
                    elif self.mine_board[x][y] == ' ': # 左击空白，无反应
                        pass
                    else: # 左击数字，处理周围八格
                        # 手动扫雷辅助型半自动扫雷
                        self._process_semi_auto(x,y)
            elif event.button == 3: # 右键
                if event.pos[0] > self.mine_sweep_rect.right or event.pos[0] < self.mine_sweep_rect.left or event.pos[1] < self.mine_sweep_rect.top or event.pos[1] > self.mine_sweep_rect.bottom:
                    return
                
                y = (event.pos[0] - self.mine_sweep_left) // (self.cell_side_length + self.cell_gap)
                x = (event.pos[1] - self.mine_sweep_top) // (self.cell_side_length + self.cell_gap)
                print(x,y,event.pos)
                
                if self.mine_cover_board[x][y] == '*': # 不能在已扫的覆盖层上标记
                    return
                if self.mine_cover_board[x][y] == '1': # 已标记，取消标记
                    self.mine_cover_board[x][y] = ' '
                    self.mine_remain_num += 1
                else: # 标记
                    if self.mine_remain_num > 0: # 只能在剩余标记数（雷数）大于0时进行标记
                        self.mine_cover_board[x][y] = '1'
                        self.mine_remain_num -= 1
        elif event.type == pygame.MOUSEBUTTONUP:
            self.clicking = False
            if event.button == 1: # 左键
                for i in range(self.tot_height):
                    for j in range(self.tot_width):
                        if self.ghost_board[i][j] == True: # 覆盖层虚化状态，还原为非虚化
                            self.ghost_board[i][j] = False

        # 在事件处理末尾标记需要自动扫雷
        if event.type in [pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP]:
            self.need_auto_sweep = True
    
    """执行手动扫雷半自动扫雷逻辑"""
    def _process_semi_auto(self, x, y):
        if self.semi_auto_mode == False:
            return
        
        # 统计标记数量
        sign_cnt = 0
        for i in range(max(0, x - 1), min(self.tot_height, x + 2)):
            for j in range(max(0, y - 1), min(self.tot_width, y + 2)):
                if self.mine_cover_board[i][j] == '1':
                    sign_cnt += 1
        # 统计雷数
        mine_num = 0
        for i in range(max(0, x - 1), min(self.tot_height, x + 2)):
            for j in range(max(0, y - 1), min(self.tot_width, y + 2)):
                if self.mine_board[i][j] == '*':
                    mine_num += 1
        if sign_cnt == mine_num: # 标记数与雷数相等
            for i in range(max(0, x - 1), min(self.tot_height, x + 2)):
                for j in range(max(0, y - 1), min(self.tot_width, y + 2)):
                    if self.mine_board[i][j] == ' ':
                        self.clear_blank(i,j)
                    if self.mine_board[i][j] == '*': # 真实层为雷
                        if self.mine_cover_board[i][j] == '1': # 覆盖层为标记
                            continue
                        self.game_over(i,j)
                        return
                    self.mine_cover_board[i][j] = '*'
        elif sign_cnt < mine_num: # 标记数小于雷数
            for i in range(max(0, x - 1), min(self.tot_height, x + 2)):
                for j in range(max(0, y - 1), min(self.tot_width, y + 2)):
                    if self.mine_cover_board[i][j] == ' ': # 覆盖层为空，覆盖层暂时虚化后还原
                        self.ghost_board[i][j] = True
    
    # 处理胜负
    def handle_win_or_lose(self):        
        if self.game_state == 'running':
            for i in range(self.tot_height):
                for j in range(self.tot_width):
                    if self.mine_board[i][j] != '*' and self.mine_cover_board[i][j] == ' ': # 真实层不是雷，覆盖层未扫，游戏未结束，返回
                        return
                    if self.mine_board[i][j] == '*' and self.mine_cover_board[i][j] == '*': # 出现覆盖层已扫的雷
                        print('game_over')
                        self.game_over()
                        return
            self.game_win() # 相等，游戏胜利
    
    def clear_blank(self,x,y):
        visited_area = [[0 for _ in range(self.tot_width)] for _ in range(self.tot_height)]
        def clear(i,j):
            if visited_area[i][j] == 1:
                return
            visited_area[i][j] = 1
            
            # 如果本格真实层是数字,只清除本格覆盖层后返回
            if self.mine_board[i][j].isdigit(): 
                self.mine_cover_board[i][j] = '*'
                return 
            
            # 本格真实层是空白，清除含本格内3*3区域格子
            clear(min(i + 1, self.tot_height - 1), j)
            clear(min(i + 1, self.tot_height - 1), min(j + 1, self.tot_width - 1))
            clear(min(i + 1, self.tot_height - 1), max(j - 1, 0))
            clear(max(i - 1, 0),j)
            clear(max(i - 1, 0), min(j + 1, self.tot_width - 1))
            clear(max(i - 1, 0), max(j - 1, 0))
            clear(i, min(j + 1, self.tot_width - 1))
            clear(i, max(j - 1, 0))
            
            self.mine_cover_board[i][j] = '*'
        clear(x,y)
    
    def game_over(self,x=-1,y=-1):
        
        if x==-1 and y==-1:
            for i in range(self.tot_height):
                for j in range(self.tot_width):
                    if self.mine_board[i][j] == '*' and self.mine_cover_board[i][j] != '1':
                        self.game_over_cell_x = i
                        self.game_over_cell_y = j
                        break
        else:
            self.game_over_cell_x=x
            self.game_over_cell_y=y
        
        self.running_state = False
        self.game_state = 'lose'
        self.mine_remain_num = self.mine_num
        
        # 显示所有雷
        for i in range(self.tot_height):
            for j in range(self.tot_width):
                if self.mine_board[i][j] == '*' and self.mine_cover_board[i][j] != '1':
                    self.mine_cover_board[i][j] = '*'
                elif self.mine_board[i][j] != '*' and self.mine_cover_board[i][j] == '1':
                    self.mine_cover_board[i][j] = '*'
        
        self.draw()
    
    def game_win(self):
        self.running_state = False
        self.mine_remain_num = self.mine_num
        self.game_state = 'win'
        
        # 去除所有真实层不为雷覆盖层空白未扫，将所有真实层为雷的覆盖层标记
        for i in range(self.tot_height):
            for j in range(self.tot_width):
                if self.mine_board[i][j] != '*' and self.mine_cover_board[i][j] == ' ':
                    self.mine_cover_board[i][j] = '*'
                if self.mine_board[i][j] == '*':
                    self.mine_cover_board[i][j] = '1'
        
        self.draw()

# 扫雷主界面
class MineSweeper():
    def __init__(self, scale = 1):
        
        self.scale = scale
        self.display_width = Config.SIZE["SCREEN_WIDTH"] * self.scale # 界面宽度
        self.display_height = Config.SIZE["SCREEN_HEIGHT"] * self.scale # 界面高度
        
        self.auto_sweep_txt = '自动扫雷：关' # 自动扫雷开关
        
        self.semi_auto_txt = '半自动辅助：开' # 手动扫雷半自动模式文字
        
        # 初始化游戏
        pygame.display.set_caption('MineSweeper')
        
        icon = assets.images['icon']  # 替换为你的图标文件路径
        pygame.display.set_icon(icon)
        
        self.screen = pygame.display.set_mode((self.display_width,self.display_height))
        self.font_headline = assets.fonts["headline"]
        self.font_auto_sweep = assets.fonts["auto_sweep"]
        self.font_semi_auto_sweep = assets.fonts["semi_auto_sweep"]
        
        self.draw_static_board()
        self.background = self.screen.copy()
    
    # 绘制静止screen
    def draw_static_board(self):
        # 背景
        self.screen.fill(Config.COLORS['background'],(0,0,self.display_width,self.display_height))
        
        # “扫雷”标题
        txt_surface = self.font_headline.render('扫雷',True,Config.COLORS['black'])
        txt_rect = txt_surface.get_rect(center=(self.display_width // 2,26))
        self.screen.blit(txt_surface,txt_rect)
    
    # 加载静止screen
    def load_static_board(self):
        self.screen.blit(self.background,(0,0))
    
    # 动态绘制
    def draw(self):
        # 自动扫雷(开/关)状态
        txt_surface = self.font_auto_sweep.render(self.auto_sweep_txt,True,Config.COLORS['black'])
        txt_rect = txt_surface.get_rect(center=(self.display_width - len(self.auto_sweep_txt) * 15,26))
        self.screen.blit(txt_surface,txt_rect)
        
        # 手动扫雷半自动扫雷辅助（开/关）状态
        txt_surface = self.font_semi_auto_sweep.render(self.semi_auto_txt,True,Config.COLORS['black'])
        txt_rect = txt_surface.get_rect(center=(self.display_width - len(self.semi_auto_txt) * 15,52))
        self.screen.blit(txt_surface,txt_rect)
    
    # 事件处理
    def handle_event(self,event,mine_area):
        if event.type == pygame.KEYDOWN: # 按键按下，注意保持英文状态
            if event.key == pygame.K_a:
                mine_area.auto_sweep_state = not mine_area.auto_sweep_state
                self.auto_sweep_txt = f'自动扫雷：{"开" if mine_area.auto_sweep_state else "关"}'
            if event.key == pygame.K_f:
                mine_area.semi_auto_mode = not mine_area.semi_auto_mode
                self.semi_auto_txt = f'半自动辅助：{"开" if mine_area.semi_auto_mode else "关"}'

if __name__  == '__main__':
    
    # 初始化pygame
    pygame.init()
    # 预加载资源
    assets.load_all()
    
    '''初始化运行类的资源'''
    mine_sweeper = MineSweeper(scale=1)
    screen = mine_sweeper.screen
    
    # 调节难度按钮
    difficulty_buttons = []
    x_positions = [Config.SIZE["SCREEN_WIDTH"] // 2 - 110, Config.SIZE["SCREEN_WIDTH"] // 2 -60, Config.SIZE["SCREEN_WIDTH"] // 2 -8, Config.SIZE["SCREEN_WIDTH"] // 2 + 44, Config.SIZE["SCREEN_WIDTH"] // 2 + 104]  # 调整坐标
    for name in Config.DIFFICULTIES.keys():
        btn = DifficultyCategory(
            center=(Config.SIZE["SCREEN_WIDTH"] // 2 + Config.DIFFICULTIES[name]["offset"], 72),
            surface=screen,
            difficulty=name
        )
        difficulty_buttons.append(btn)
    
    # 扫雷区域
    mine_area = MineArea(surface=screen)
    # 重开游戏按钮
    restart = Restart(surface=screen,rect=mine_area.restart_rect)
    # 自定义模式下调节宽、高、雷数按钮
    button_width = CustomButton('宽',screen,init_num=mine_area.tot_width,left=Config.SIZE["SCREEN_WIDTH"] // 2 - 96 - 22 - 3 - 96)
    button_height = CustomButton('高',screen,init_num=mine_area.tot_height,left=Config.SIZE["SCREEN_WIDTH"] // 2 - 96)
    button_mine_num = CustomButton('雷',screen,init_num=mine_area.mine_num,left=Config.SIZE["SCREEN_WIDTH"] // 2 + 3 + 22)
    button_confirm = ConfirmButton(screen,left=Config.SIZE["SCREEN_WIDTH"] // 2 + 3 + 22 + 96 + 10)
    # 自定义模式控制
    custom = Custom(button_width,button_height,button_mine_num,button_confirm)
    
    # pygame帧率钟
    clock = pygame.time.Clock()
    
    # 游戏运行时
    while True:
        # Quit game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            mine_sweeper.handle_event(event,mine_area)
            
            button_width.handle_event(event)
            button_height.handle_event(event)
            button_mine_num.handle_event(event)
            button_confirm.handle_event(event)
            
            restart.handle_event(event,mine_area)
            
            custom.handle_event(mine_area)
            
            for btn in difficulty_buttons:
                btn.handle_event(event, mine_area)  # 注意要传递 mine_area 参数
            
            mine_area.handle_event(event)
        
        # 处理计时
        mine_area.handle_time()
        
        # 处理胜负
        mine_area.handle_win_or_lose()
        
        # 自动扫雷
        mine_area.auto_sweep()
        
        # restart处理胜负
        restart.handle_game_state(mine_area)
        
        # 加载静止界面
        mine_sweeper.load_static_board()
        
        # 加载动态界面
        mine_sweeper.draw()
        
        # 难度按键
        for btn in difficulty_buttons:
            btn.draw()
        
        # 自定义按钮
        button_width.draw(mine_area)
        button_height.draw(mine_area)
        button_mine_num.draw(mine_area)
        button_confirm.draw(mine_area)
        
        # 加载雷区画面
        mine_area.draw()
        
        # restart 按钮
        restart.draw(mine_area.restart_rect)
        
        # 刷新screen
        pygame.display.flip()
        # 游戏帧率
        clock.tick(165)