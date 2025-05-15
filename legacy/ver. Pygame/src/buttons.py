# buttons.py

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