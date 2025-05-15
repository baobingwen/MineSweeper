import json
from pathlib import Path
from config import Config

class Records:
    def __init__(self):
        self.file_path = Config.RECORDS_FILE
        self.data = self._load_data()
        
        # 初始化数据结构
        if 'records' not in self.data:
            self.data['records'] = {}
            self._save_records()

    def _load_data(self):
        """加载记录文件，若不存在则创建默认"""
        try:
            if not self.file_path.exists():
                self._create_default_records()
            
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载记录失败: {e}，将创建新文件")
            return self._create_default_records()

    def _create_default_records(self):
        """创建默认记录结构"""
        default = {
            "records": {},  # 用于存储按难度、辅助扫雷、自动扫雷分类的记录
            "settings": {   # 用于存储用户设置
                "auto_sweep": False,
                "semi_auto": True
            }
        }
        self._save_records(default)
        return default

    def _save_records(self, data=None):
        """保存记录到文件"""
        if data is None:
            data = self.data
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def add_record(self, difficulty, auto_sweep, semi_auto, time):
        """
        添加一条新的记录
        :param difficulty: 难度（如"基础"、"中级"等）
        :param auto_sweep: 是否开启自动扫雷（True/False）
        :param semi_auto: 是否开启辅助扫雷（True/False）
        :param time: 游戏时间（秒）
        """
        
        key = f"{difficulty}|{auto_sweep}|{semi_auto}"
        if key not in self.data['records']:
            self.data['records'][key] = []
        
        # 保留前10条记录并按时间排序
        if time in self.data['records'][key]: # 如果已经有记录，排除（在记录记录保持者时此优化可省去）
            return
        self.data['records'][key].append(time)
        self.data['records'][key].sort()
        self.data['records'][key] = self.data['records'][key][:10]
        self._save_records()

    def get_best_time(self, difficulty, auto_sweep, semi_auto):
        """
        获取当前配置的最佳时间
        :param difficulty: 难度
        :param auto_sweep: 是否开启自动扫雷
        :param semi_auto: 是否开启辅助扫雷
        :return: 最佳时间（秒），如果没有记录则返回None
        """
        
        key = f"{difficulty}|{auto_sweep}|{semi_auto}"
        if key in self.data['records'] and self.data['records'][key]:
            return min(self.data['records'][key])
        return None
    
    def get_all_records(self, difficulty, auto_sweep, semi_auto):
        """
        获取当前配置的所有记录
        :param difficulty: 难度
        :param auto_sweep: 是否开启自动扫雷
        :param semi_auto: 是否开启辅助扫雷
        :return: 当前配置的所有记录（列表）
        """
        
        key = f"{difficulty}|{auto_sweep}|{semi_auto}"
        return self.data['records'].get(key, [])

    def update_settings(self, auto_sweep, semi_auto):
        """
        更新用户设置
        :param auto_sweep: 是否开启自动扫雷，一旦是则一直是
        :param semi_auto: 是否开启辅助扫雷，一旦是则一直是
        """
        self.data['settings']['auto_sweep'] = auto_sweep
        self.data['settings']['semi_auto'] = semi_auto
        self._save_records()

    def get_settings(self):
        """
        获取当前用户设置
        :return: 包含auto_sweep和semi_auto的字典
        """
        return self.data['settings']

# 全局记录实例
records = Records()