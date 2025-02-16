# config.py
from pathlib import Path

class Config:
    # ---------- 颜色 ----------
    COLORS = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (255, 0, 0),
        "gray": (134, 134, 134),
        
        "num_1": (0, 0, 255),
        "num_2": (0, 128, 0),
        "num_3": (255,0,0),
        "num_4": (0,0,128),
        "num_5": (128,0,0),
        "num_6": (0,128,128),
        "num_7": (0,0,0),
        "num_8": (128,128,128),
        
        "background": (247, 247, 240),
        "mine_area": (204, 204, 204),
        "mine_area_dark": (124, 124, 124),
        
        "mine_center": (192,192,192),
        "mine_light": (254,254,254),
        "mine_dark": (121,121,121),
        "mine_hidden_background": (190,190,190),

        "tri_button_shadow": (139,139,139),
        "tri_button_dark": (99,99,99)
    }

    # ---------- 字体 ----------
    FONTS = {
        # 基础字体族（按用途分类）
        "families": {
            "default": "华文中宋",
            "monospace": "Consolas",
            "digital": "DS-Digital",
            "title": "华文中宋"
        },
        
        # 字体样式预设（按组件/场景分类）
        "presets": {
            "headline": {
                "family": "title",
                "size": 32,
            },
            "button_difficulty": {
                "family": "default",
                "size": 20,
            },
            "num_remaining_flags": {
                "family": "digital",
                "size": 25,
            },
            "num_timer": {
                "family": "digital",
                "size": 30,
            },
            "num_custom_button": {
                "family": "monospace",
                "size": 30
            },
            "name_custom_button": {
                "family": "default",
                "size": 20,
            },
            "auto_sweep": {
                "family": "default",
                "size": 20,
            },
            "semi_auto_sweep": {
                "family": "default",
                "size": 20,
            },
            "debug": {  # 示例：调试信息的字体
                "family": "monospace",
                "size": 12,
                "color": (255, 0, 0)  # 扩展支持颜色（需适配加载逻辑）
            }
        }        
    }

    # ---------- 尺寸 ----------
    SIZE = {
        "CELL_SIDE_LENGTH": 25,     # 单元格边长
        "CELL_GAP": 2,              # 单元格间距
        "CELL_BORDER": 9,           # 雷区边框宽度
        
        "SCREEN_WIDTH": 984,        # 界面宽度
        "SCREEN_HEIGHT": 858,       # 界面高度
        
        "CUSTOM_BUTTON_WIDTH": 96,  # 自定义框宽度
        "CUSTOM_BUTTON_HEIGHT": 36, # 自定义框高度
        "CUSTOM_BUTTON_TOP": 100,   # 自定义框top
    }

    # ---------- 图片路径 ----------
    IMAGE_DIR = Path("./image")
    IMAGE_PATHS = {
        # restart button
        "restart_normal": IMAGE_DIR / "face0.png",
        "restart_normal_down": IMAGE_DIR / "face1.png",
        "restart_win": IMAGE_DIR / "face4.png",
        "restart_win_down": IMAGE_DIR / "face9.png",
        "restart_lose": IMAGE_DIR / "face3.png",
        "restart_lose_down": IMAGE_DIR / "face8.png",
        "restart_clicking": IMAGE_DIR / "face2.png",
        
        # mine area
        "hidden_1": IMAGE_DIR / "1.png",
        "hidden_2": IMAGE_DIR / "2.png",
        "hidden_3": IMAGE_DIR / "3.png",
        "hidden_4": IMAGE_DIR / "4.png",
        "hidden_5": IMAGE_DIR / "5.png",
        "hidden_6": IMAGE_DIR / "6.png",
        "hidden_7": IMAGE_DIR / "7.png",
        "hidden_8": IMAGE_DIR / "8.png",
        "hidden_mine_normal": IMAGE_DIR / "mine0.png",
        "hidden_mine_triggar": IMAGE_DIR / "mine2.png",
        
        "cover_blank": IMAGE_DIR / "blank.png",
        "cover_flag": IMAGE_DIR / "flag.png",
        
        # Mine Sweeper
        "icon": IMAGE_DIR / "icon.ico"
    }

    # ---------- 难度预设 ----------
    DIFFICULTIES = {
        "基础": {"width": 9, "height": 9, "mines": 10, "offset": -110},
        "中级": {"width": 16, "height": 16, "mines": 40, "offset": -60},
        "专家": {"width": 30, "height": 16, "mines": 99, "offset": -8},
        "满屏": {"width": 35, "height": 19, "mines": 137, "offset": 44},
        "自定义": {"offset": 104}
    }