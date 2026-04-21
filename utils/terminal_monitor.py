"""
终端监视器 - 基于单例模式的固定位置终端监视器
支持实时参数监测和事件追踪
"""

import threading
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import deque
import yaml
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.live import Live



class TerminalMonitor:
    """单例监视器，在终端下方 2/3 区域显示参数和事件。
    
    功能：
    - 实时参数显示（key:value 对，tab 分隔）
    - 事件追踪（带时间戳、事件名、描述）
    - 后台异步更新
    - 线程安全
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化监视器。
        
        参数：
            config_path: monitor_config.yaml 的路径。如果为 None，使用默认路径。
        """
        self._initialized = False
        
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "monitor_config.yaml"
        else:
            config_path = Path(config_path)
        
        self._config = self._load_config(config_path)
        
        self._state_lock = threading.Lock()
        self._params: Dict[str, Any] = {}
        self._events: deque = deque(maxlen=self._config['monitor']['max_events'])
        self._last_render_hash = None
        
        self._console = Console()
        self._live: Optional[Live] = None
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        self._initialized = True
    
    @classmethod
    def get_instance(cls, config_path: Optional[str] = None) -> "TerminalMonitor":
        """获取单例实例（双检锁）。
        
        参数：
            config_path: 可选的配置文件路径
            
        返回：
            单例 TerminalMonitor 实例
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config_path)
        return cls._instance
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """加载 YAML 配置文件。
        
        参数：
            config_path: 配置文件路径
            
        返回：
            配置字典
        """
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None or 'monitor' not in config:
            raise ValueError("无效的监视器配置文件格式")
        
        return config
    
    def start(self) -> None:
        """启动后台渲染线程。"""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """停止后台渲染线程。"""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._live is not None:
            self._live.stop()
            self._live = None
        
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
    
    def is_running(self) -> bool:
        """检查监视器是否正在运行。"""
        return self._running
    
    def set_param(self, key: str, value: Any) -> None:
        """设置单个参数。
        
        参数：
            key: 参数名
            value: 参数值
        """
        with self._state_lock:
            self._params[key] = value
    
    def update_params(self, params: Dict[str, Any]) -> None:
        """批量更新参数。
        
        参数：
            params: 参数字典
        """
        with self._state_lock:
            self._params.update(params)
    
    def get_params(self) -> Dict[str, Any]:
        """获取当前参数（快照）。
        
        返回：
            参数字典副本
        """
        with self._state_lock:
            return dict(self._params)
    
    def clear_params(self) -> None:
        """清除所有参数。"""
        with self._state_lock:
            self._params.clear()
    
    def add_event(self, event_name: str, description: str = "") -> None:
        """添加事件记录。
        
        参数：
            event_name: 事件名称
            description: 事件描述（可选）
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        event_entry = (timestamp, event_name, description)
        
        with self._state_lock:
            self._events.append(event_entry)
    
    def get_events(self) -> List[tuple]:
        """获取当前事件列表（快照）。
        
        返回：
            事件列表副本
        """
        with self._state_lock:
            return list(self._events)
    
    def _get_state_hash(self) -> int:
        """计算当前状态的哈希值以检测变化。
        
        返回：
            状态哈希值
        """
        with self._state_lock:
            params_repr = tuple(sorted(self._params.items()))
            events_repr = tuple(self._events)
            
            return hash((params_repr, events_repr))
    
    def _render_loop(self) -> None:
        """后台线程循环，定期渲染监视器。"""
        refresh_rate = self._config['monitor']['refresh_rate_ms'] / 1000.0
        
        while not self._stop_event.is_set():
            current_hash = self._get_state_hash()
            
            if self._last_render_hash != current_hash or self._live is None:
                self._render()
                self._last_render_hash = current_hash
            
            time.sleep(refresh_rate)
    
    def _render(self) -> None:
        """渲染监视器布局并更新显示。"""
        try:
            layout = self._build_layout()
            
            if self._live is None:
                self._live = Live(layout, console=self._console, refresh_per_second=60)
                self._live.start()
            else:
                self._live.update(layout)
        except Exception:
            pass
    
    def _build_layout(self) -> Layout:
        """构建监视器布局（参数区和事件区）。
        
        返回：
            Rich Layout 对象
        """
        layout = Layout()
        
        layout.split(
            Layout(name="param_panel", minimum_size=2),
            Layout(name="event_panel", minimum_size=2),
            ratio=[1, 1]
        )
        
        layout["param_panel"].update(self._build_param_panel())
        layout["event_panel"].update(self._build_event_panel())
        
        return layout
    
    def _build_param_panel(self) -> Panel:
        """构建参数显示面板。
        
        返回：
            显示参数的 Panel
        """
        with self._state_lock:
            if not self._params:
                content = Text("（无参数）", style="dim")
            else:
                param_items = []
                for key, value in self._params.items():
                    value_str = str(value)
                    if len(value_str) > 20:
                        value_str = value_str[:17] + "..."
                    param_items.append(f"{key}:{value_str}")
                
                content_str = "\t".join(param_items)
                content = Text(content_str, style="green")
        
        return Panel(content, title="[bold]参数[/bold]", border_style="green")
    
    def _build_event_panel(self) -> Panel:
        """构建事件显示面板。
        
        返回：
            显示事件的 Panel
        """
        content_lines = []
        
        with self._state_lock:
            if self._events:
                for timestamp, event_name, description in self._events:
                    timestamp_text = Text(f"[{timestamp}]", style="dim")
                    event_text = Text(f" {event_name:12}", style="bold cyan")
                    desc_text = Text(f" {description}" if description else "")
                    
                    line = timestamp_text + event_text + desc_text
                    content_lines.append(line)
            else:
                content_lines.append(Text("（无事件）", style="dim"))
        
        from rich.console import Group
        content = Group(*content_lines) if content_lines else Text("（无事件）", style="dim")
        
        return Panel(content, title="[bold]事件[/bold]", border_style="yellow")
