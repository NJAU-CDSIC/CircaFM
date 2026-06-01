import logging

class CustomLogger:
    def __init__(self, log_file: str, log_level: str = 'INFO', encoding='utf-8'):
        """
        初始化日志器并直接支持日志记录。
        
        Args:
            log_file (str): 日志文件路径
            log_level (str): 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        # 获取日志器
        self.logger = logging.getLogger()
        
        # 设置日志级别
        self.logger.setLevel(getattr(logging, log_level.upper()))  # 使用 getattr 来获取日志级别

        # 创建文件处理器
        self.file_handler = logging.FileHandler(log_file, encoding=encoding)
        self.file_handler.setLevel(getattr(logging, log_level.upper()))  # 设置处理器日志级别
        
        # 创建日志格式
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 设置日志格式
        self.file_handler.setFormatter(self.formatter)
        
        # 将文件处理器添加到日志器
        self.logger.addHandler(self.file_handler)

    def log(self, level: str, message: str):
        """
        记录日志
        
        Args:
            level (str): 日志级别 ('debug', 'info', 'warning', 'error', 'critical')
            message (str): 日志消息
        """
        # 获取日志级别的对应方法
        log_method = getattr(self.logger, level.lower())
        log_method(message)

    # 以下方法使得可以直接通过实例调用相应的日志方法
    def info(self, message: str):
        self.log('info', message)

    def debug(self, message: str):
        self.log('debug', message)

    def warning(self, message: str):
        self.log('warning', message)

    def error(self, message: str):
        self.log('error', message)

    def critical(self, message: str):
        self.log('critical', message)