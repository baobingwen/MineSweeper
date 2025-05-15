# mine_view.py
import pygame
from config import Config
from assets import assets
from typing import Tuple

class MineView:
    def __init__(self, logic, surface: pygame.Surface) -> None:
        """
        扫雷游戏视图模块
        
        :param logic: MineLogic实例
        :param surface: 绘制目标Surface
        """
        # 预加载资源
        assets.load_all()
        
        self.logic = logic
        self.surface = surface
        
        # 动态布局参数
        self.cell_size = Config.SIZE["CELL_SIDE_LENGTH"]
        self.cell_gap = Config.SIZE["CELL_GAP"]
        self.border_width = Config.SIZE["CELL_BORDER"]
        
        # 颜色配置
        self.colors = Config.COLORS
        
        # 资源缓存
        self._load_resources()
        
        # 初始化布局
        self._calculate_layout()
    
    def _load_resources(self) -> None:
        """预加载所有图形资源"""
        self.images_hidden_number = {
            '1': assets.images['hidden_1'],
            '2': assets.images['hidden_2'],
            '3': assets.images['hidden_3'],
            '4': assets.images['hidden_4'],
            '5': assets.images['hidden_5'],
            '6': assets.images['hidden_6'],
            '7': assets.images['hidden_7'],
            '8': assets.images['hidden_8'],
        }
        self.images_mine = {
            'normal': assets.images['hidden_mine_normal'],
            'triggered': assets.images['hidden_mine_triggar'],
        }
        self.cover_images = {
            'blank': assets.images['cover_blank'],
            'flag': assets.images['cover_flag'],
        }
        
        # 字体资源
        self.mine_num_font = assets.fonts["num_remaining_flags"]
        self.time_font = assets.fonts["num_timer"]
    
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