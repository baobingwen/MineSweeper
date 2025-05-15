# game_controller.py

class GameController:
    def handle_event(self,event):
        if self.game_state == 'lose' or self.game_state == 'win':
            return
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            # 雷区事件
            if event.button == 1: # 左键
                
                # 超出扫雷区域，无反应
                if event.pos[0] > self.mine_sweep_rect.right or event.pos[0] < self.mine_sweep_rect.left or event.pos[1] < self.mine_sweep_rect.top or event.pos[1] > self.mine_sweep_rect.bottom:
                    return
                
                y = (event.pos[0] - self.mine_sweep_left) // (self.cell_side_length + self.cell_gap)
                x = (event.pos[1] - self.mine_sweep_top) // (self.cell_side_length + self.cell_gap)
                print(x,y,event.pos)
                
                # 检查坐标是否在有效范围内
                if x < 0 or x >= len(self.mine_cover_board) or y < 0 or y >= len(self.mine_cover_board[0]):
                    return
                
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

        # 本局鼠标点击次数
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.click_cnt += 1  # 总点击+1
            if event.button == 1: # 左键
                self.left_click_cnt += 1
            elif event.button == 3: # 右键
                self.right_click_cnt += 1

        # 在事件处理末尾标记需要自动扫雷
        if event.type in [pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP]:
            self.need_auto_sweep = True

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

    def handle_time(self):
        if self.game_state == 'running': # 游戏已经开始
            self.now_time = str(round((pygame.time.get_ticks() - self.start_tick) / 1000,1)) # 现在显示时间