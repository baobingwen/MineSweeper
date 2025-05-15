# game.py

# 扫雷主界面
class MineSweeper():
    def __init__(self, scale = 1):
        
        self.scale = scale
        self.display_width = Config.SIZE["SCREEN_WIDTH"] * self.scale # 界面宽度
        self.display_height = Config.SIZE["SCREEN_HEIGHT"] * self.scale # 界面高度
        
        self.auto_sweep_txt = '自动扫雷：关' # 自动扫雷开关
        self.semi_auto_txt = '半自动辅助：开' # 手动扫雷半自动模式文字
        self.click_cnt_txt = '本局鼠标点击次数：' # 本局鼠标点击次数文字
        self.left_click_cnt_txt = '本局鼠标左击次数：' # 本局鼠标左击次数文字
        self.right_click_cnt_txt = '本局鼠标右击次数：' # 本局鼠标右击次数文字
        
        # 初始化游戏
        pygame.display.set_caption('MineSweeper')
        
        icon = assets.images['icon']
        pygame.display.set_icon(icon)
        
        self.screen = pygame.display.set_mode((self.display_width,self.display_height))
        self.font_headline = assets.fonts["headline"]
        self.font_auto_sweep = assets.fonts["auto_sweep"]
        self.font_semi_auto_sweep = assets.fonts["semi_auto_sweep"]
        self.font_click_cnt = assets.fonts["num_click"]
        self.font_time_record = assets.fonts["time_record"]
        
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
        txt_rect = txt_surface.get_rect(center=(self.display_width - len(self.auto_sweep_txt) * 13,26))
        self.screen.blit(txt_surface,txt_rect)
        
        # 手动扫雷半自动扫雷辅助（开/关）状态
        txt_surface = self.font_semi_auto_sweep.render(self.semi_auto_txt,True,Config.COLORS['black'])
        txt_rect = txt_surface.get_rect(center=(self.display_width - len(self.semi_auto_txt) * 13,26*2))
        self.screen.blit(txt_surface,txt_rect)
        
        # 鼠标点击次数
        txt_surface = self.font_click_cnt.render(f'{self.click_cnt_txt}{mine_area.click_cnt}',True,Config.COLORS['black'])
        txt_rect = txt_surface.get_rect(center=(len(f'{self.click_cnt_txt}{mine_area.click_cnt}') * 8,26))
        self.screen.blit(txt_surface,txt_rect)
        # 左击
        txt_surface = self.font_click_cnt.render(f'{self.left_click_cnt_txt}{mine_area.left_click_cnt}',True,Config.COLORS['black'])
        txt_rect = txt_surface.get_rect(center=(len(f'{self.left_click_cnt_txt}{mine_area.left_click_cnt}') * 8,52))
        self.screen.blit(txt_surface,txt_rect)
        # 右击
        txt_surface = self.font_click_cnt.render(f'{self.right_click_cnt_txt}{mine_area.right_click_cnt}',True,Config.COLORS['black'])
        txt_rect = txt_surface.get_rect(center=(len(f'{self.right_click_cnt_txt}{mine_area.right_click_cnt}') * 8,78))
        self.screen.blit(txt_surface,txt_rect)
        
        # 最佳时间显示
        from records import records
        best_time = records.get_best_time(
            mine_area.difficulty,
            mine_area.auto_sweep_used,
            mine_area.semi_auto_used
        )
        best_time_txt = f"当前模式最佳: {best_time:.1f}s" if best_time is not None else "当前模式最佳: 暂无"
        txt_surface = self.font_time_record.render(best_time_txt, True, Config.COLORS['black'])
        txt_rect = txt_surface.get_rect(center=(self.display_width - 150, 84))
        self.screen.blit(txt_surface, txt_rect)
    
    # 事件处理
    def handle_event(self,event,mine_area):
        if event.type == pygame.KEYDOWN: # 按键按下，注意保持英文状态
            if event.key == pygame.K_a:
                from records import records
                mine_area.auto_sweep_state = not mine_area.auto_sweep_state
                
                if mine_area.auto_sweep_state:
                    mine_area.auto_sweep_used = True  # 标记自动扫雷被启用过
                self.auto_sweep_txt = f'自动扫雷：{"开" if mine_area.auto_sweep_state else "关"}'
                #records.update_settings(mine_area.auto_sweep_state, mine_area.semi_auto_mode)
            if event.key == pygame.K_f:
                from records import records
                mine_area.semi_auto_mode = not mine_area.semi_auto_mode
                if mine_area.auto_sweep_state:
                    mine_area.semi_auto_used = True  # 标记自动扫雷被启用过
                self.semi_auto_txt = f'半自动辅助：{"开" if mine_area.semi_auto_mode else "关"}'
                #records.update_settings(mine_area.auto_sweep_state, mine_area.semi_auto_mode)
            elif event.key == pygame.K_r:  # 按R键查看当前模式所有记录
                from records import records
                current_records = records.get_all_records(
                    mine_area.difficulty,
                    mine_area.auto_sweep_used,
                    mine_area.semi_auto_used
                )
                print(f"当前模式（{mine_area.difficulty},自动扫雷：{mine_area.auto_sweep_used},辅助扫雷：{mine_area.semi_auto_used}）历史记录：")
                for i, t in enumerate(current_records, 1):
                    print(f"{i}. {t:.1f}秒")

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