# assets.py
import sys
import pygame
from config import Config

class Assets:
    def __init__(self):
        self.images = {}
        self.fonts = {}
    
    def load_all(self):
        """显式调用加载资源，确保 pygame 已初始化"""
        self.load_images()
        self.load_fonts()

    def load_images(self):
        """加载所有图片"""
        for name, path in Config.IMAGE_PATHS.items():
            try:
                self.images[name] = pygame.image.load(str(path))
            except FileNotFoundError:
                print(f"错误：图片 {path} 未找到！")
                sys.exit(1)
    
    def load_fonts(self):
        """加载所有字体"""
        for preset_name, preset_config in Config.FONTS["presets"].items():
            family_key = preset_config.get("family", "default")
            font_name = Config.FONTS["families"].get(family_key, Config.FONTS["families"]["default"])
            font_size = preset_config.get("size", 20)
            bold = preset_config.get("bold", False)
            italic = preset_config.get("italic", False)
            
            try:
                # 加载字体
                self.fonts[preset_name] = pygame.font.SysFont(
                    font_name, font_size, bold, italic
                )
            except Exception as e:
                print(f"字体加载失败：{preset_name} - {e}")
                # 回退到默认字体
                self.fonts[preset_name] = pygame.font.SysFont(None, font_size)

# 全局资源实例（但需显式调用 load_all()）
assets = Assets()