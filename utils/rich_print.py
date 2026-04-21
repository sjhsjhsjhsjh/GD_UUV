from rich.console import Console
from rich.text import Text
from datetime import datetime

LEVELS = ["LOG", "WARN", "ERROR", "DEBUG", "INFO"]
LEVEL_COLORS = {
    "LOG": "green",
    "WARN": "yellow",
    "ERROR": "red",
    "DEBUG": "cyan",
    "INFO": "blue",
    "SUCC": "green",
    "FAIL": "red",
}

MAX_PREFIX_LEN = max(len(f"[{l}]:") for l in LEVELS)

# 全局console对象
_console = Console()


def log(console: Console, level: str, message: str):
    """
    使用 rich 库在控制台打印带有时间戳和颜色的日志消息。

    :param console: rich Console 对象
    :param level: 日志级别（LOG, WARN, ERROR, DEBUG, INFO）
    :param message: 要打印的日志消息
    """
    time_str = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{level}]:"
    spaces = " " * (MAX_PREFIX_LEN - len(prefix) + 1)  # 冒号后对齐间距

    text = Text()
    text.append(f"[{time_str}] ", style="dim")
    text.append(prefix, style=f"bold {LEVEL_COLORS.get(level, 'white')}")
    text.append(spaces)
    text.append(message)

    console.print(text)


# ===== 便捷包装函数 =====

def print_info(message: str, end: str = "\n"):
    """
    打印信息级别的日志（蓝色）
    
    参数：
        message: str
            要打印的消息
        end: str
            结尾字符，默认为换行符
    """
    time_str = datetime.now().strftime("%H:%M:%S")
    prefix = "[INFO]:"
    spaces = " " * (MAX_PREFIX_LEN - len(prefix) + 1)
    
    text = Text()
    text.append(f"[{time_str}] ", style="dim")
    text.append(prefix, style=f"bold {LEVEL_COLORS['INFO']}")
    text.append(spaces)
    text.append(message)
    
    _console.print(text, end=end)


def print_warn(message: str, end: str = "\n"):
    """
    打印警告级别的日志（黄色）
    
    参数：
        message: str
            要打印的消息
        end: str
            结尾字符，默认为换行符
    """
    time_str = datetime.now().strftime("%H:%M:%S")
    prefix = "[WARN]:"
    spaces = " " * (MAX_PREFIX_LEN - len(prefix) + 1)
    
    text = Text()
    text.append(f"[{time_str}] ", style="dim")
    text.append(prefix, style=f"bold {LEVEL_COLORS['WARN']}")
    text.append(spaces)
    text.append(message)
    
    _console.print(text, end=end)


def print_error(message: str, end: str = "\n"):
    """
    打印错误级别的日志（红色）
    
    参数：
        message: str
            要打印的消息
        end: str
            结尾字符，默认为换行符
    """
    time_str = datetime.now().strftime("%H:%M:%S")
    prefix = "[ERROR]:"
    spaces = " " * (MAX_PREFIX_LEN - len(prefix) + 1)
    
    text = Text()
    text.append(f"[{time_str}] ", style="dim")
    text.append(prefix, style=f"bold {LEVEL_COLORS['ERROR']}")
    text.append(spaces)
    text.append(message)
    
    _console.print(text, end=end)


def print_debug(message: str, end: str = "\n"):
    """
    打印调试级别的日志（青色）
    
    参数：
        message: str
            要打印的消息
        end: str
            结尾字符，默认为换行符
    """
    time_str = datetime.now().strftime("%H:%M:%S")
    prefix = "[DEBUG]:"
    spaces = " " * (MAX_PREFIX_LEN - len(prefix) + 1)
    
    text = Text()
    text.append(f"[{time_str}] ", style="dim")
    text.append(prefix, style=f"bold {LEVEL_COLORS['DEBUG']}")
    text.append(spaces)
    text.append(message)
    
    _console.print(text, end=end)


def print_success(message: str, end: str = "\n"):
    """
    打印成功级别的日志（绿色）
    
    参数：
        message: str
            要打印的消息
        end: str
            结尾字符，默认为换行符
    """
    time_str = datetime.now().strftime("%H:%M:%S")
    prefix = "[SUCC]:"
    spaces = " " * (MAX_PREFIX_LEN - len(prefix) + 1)
    
    text = Text()
    text.append(f"[{time_str}] ", style="dim")
    text.append(prefix, style=f"bold {LEVEL_COLORS['SUCC']}")
    text.append(spaces)
    text.append(message)
    
    _console.print(text, end=end)
