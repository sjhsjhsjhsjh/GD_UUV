#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习 Step 级日志分析和回放工具

功能说明：
1. count-games 模式：快速定位日志中包含的游戏轮数
2. replay-episode 模式：根据日志记录回放单轮游戏，采集详细数据

用法：
    python step_log_analyzer.py count-games <log_file_path>
    python step_log_analyzer.py replay-episode <log_file_path> <episode_index>
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from omegaconf import OmegaConf

# 导入项目模块
from utils.rich_print import print_info, print_error, print_success, print_warn
from env.env import Env


class StepLogAnalyzer:
    """Step 级日志分析器
    
    该类负责解析和回放强化学习的step级日志文件。
    日志格式：
        第1行（偶数行）：初始状态 -> {uuv_x},{uuv_y},{uuv_z},{enemy_y},{enemy_forward_direction}
        第2行（奇数行）：action序列 -> 多个数字[0-6]，以#结尾表示游戏结束
    """

    def __init__(self, log_file_path: str):
        """初始化分析器
        
        参数：
            log_file_path: step日志文件路径
        """
        self.log_file_path = Path(log_file_path)
        if not self.log_file_path.exists():
            print(f"[ERROR] 日志文件不存在: {self.log_file_path}", file=sys.stderr)
            raise FileNotFoundError(f"日志文件不存在: {self.log_file_path}")
        
        self.episodes = []  # 存储解析后的所有episode数据
        self._parse_log_file()
    
    def _parse_log_file(self):
        """解析日志文件，提取所有episode的初始状态和action序列
        
        日志文件格式：
            - 奇数行（索引为偶数）：初始状态，格式为 x,y,z,enemy_y,direction
            - 偶数行（索引为奇数）：action序列，以#结尾
        """
        try:
            with open(self.log_file_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[ERROR] 无法读取日志文件: {e}", file=sys.stderr)
            raise
        
        if len(lines) < 2:
            print(f"[WARN] 日志文件行数过少，无法解析", file=sys.stderr)
            return
        
        i = 0
        episode_index = 0
        while i < len(lines) - 1:
            # 读取初始状态行
            state_line = lines[i].strip()
            if not state_line:
                i += 1
                continue
            
            # 读取action序列行
            action_line = lines[i + 1].strip()
            
            try:
                # 解析初始状态：x,y,z,enemy_y,direction
                state_parts = state_line.split(',')
                if len(state_parts) != 5:
                    print(f"[WARN] 第{i+1}行状态格式错误: {state_line}", file=sys.stderr)
                    i += 2
                    continue
                
                uuv_x, uuv_y, uuv_z, enemy_y, enemy_direction = map(int, state_parts)
                
                # 解析action序列：提取所有0-6的数字，以#结尾
                actions = []
                for char in action_line:
                    if char in '0123456':
                        actions.append(int(char))
                    elif char == '#':
                        break
                
                self.episodes.append({
                    'index': episode_index,
                    'uuv_x': uuv_x,
                    'uuv_y': uuv_y,
                    'uuv_z': uuv_z,
                    'enemy_y': enemy_y,
                    'enemy_direction': enemy_direction,
                    'actions': actions,
                    'action_count': len(actions)
                })
                episode_index += 1
            
            except Exception as e:
                print(f"[WARN] 解析第{i+1}行数据时出错: {e}", file=sys.stderr)
            
            i += 2
    
    def count_games(self) -> int:
        """统计日志中的游戏轮数
        
        返回：
            游戏总轮数
        """
        return len(self.episodes)
    
    def get_episode(self, episode_index: int) -> Optional[dict]:
        """获取指定episode的数据
        
        参数：
            episode_index: episode索引（0-based）
        
        返回：
            episode数据字典，或None如果索引不合法
        """
        if 0 <= episode_index < len(self.episodes):
            return self.episodes[episode_index]
        return None
    
    def get_episodes_data(self):
        """获取所有episode的数据，返回结构化格式
        
        返回：
            包含所有episode信息的列表
        """
        result = []
        for ep in self.episodes:
            result.append({
                'index': ep['index'],
                'uuv_x': ep['uuv_x'],
                'uuv_y': ep['uuv_y'],
                'uuv_z': ep['uuv_z'],
                'enemy_y': ep['enemy_y'],
                'enemy_direction': ep['enemy_direction'],
                'action_count': ep['action_count']
            })
        return result


def _convert_reward_details(reward_details: dict) -> dict:
    """将奖励分量的中文字段名转换为英文
    
    参数：
        reward_details: 包含中文字段名的奖励分量字典
    
    返回：
        包含英文字段名的奖励分量字典
    """
    if not reward_details:
        return {}
    
    # 字段名映射
    field_mapping = {
        'sum_隐蔽奖励': 'sum_stealth_reward',
        'sum_靠近奖励': 'sum_approach_reward',
        'sum_TL梯度奖励': 'sum_tl_gradient_reward',
        'sum_范围平均TL奖励': 'sum_area_average_tl_reward',
        'sum_固定时间惩罚': 'sum_fixed_time_penalty'
    }
    
    converted = {}
    for key, value in reward_details.items():
        new_key = field_mapping.get(key, key)
        converted[new_key] = value
    
    return converted


def load_config(config_path: str = "configs/main_config.yaml") -> object:
    """加载配置文件
    
    参数：
        config_path: 配置文件路径
    
    返回：
        DictConfig对象
    """
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"[ERROR] 配置文件不存在: {cfg_path}", file=sys.stderr)
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    
    return OmegaConf.load(cfg_path)


def replay_episode(analyzer: StepLogAnalyzer, episode_index: int, cfg: object, output_dir: Optional[Path] = None):
    """回放单个episode，采集详细数据
    
    参数：
        analyzer: StepLogAnalyzer实例
        episode_index: 要回放的episode索引
        cfg: 环境配置对象
        output_dir: 可选的输出目录
    
    返回：
        包含回放结果的字典，或 None 如果失败
    """
    import io
    
    episode_data = analyzer.get_episode(episode_index)
    if episode_data is None:
        print(f"[ERROR] Episode索引越界: {episode_index}，总共有 {analyzer.count_games()} 轮游戏", file=sys.stderr)
        return None
    
#     print(f"[INFO] 开始回放 Episode #{episode_index}", file=sys.stderr)
    
    # 保存原始 stdout
    old_stdout = sys.stdout
    
    try:
        # 创建环境实例时，重定向 stdout 以避免污染输出
        sys.stdout = io.StringIO()
        
        try:
            # 禁用TCP服务器并创建环境实例
            env = Env(cfg, output_dir=output_dir, enable_step_log_server=False)
        finally:
            # 恢复 stdout
            sys.stdout = old_stdout
        
        # print(f"[INFO] 环境初始化完成", file=sys.stderr)
        
        # 重置环境
        env.reset()
        
        # 覆盖初始位置为日志中记录的位置
        env.uuv.x = episode_data['uuv_x']
        env.uuv.y = episode_data['uuv_y']
        env.uuv.z = episode_data['uuv_z']
        env.enemy.y = episode_data['enemy_y']
        env.enemy_forward_direction = episode_data['enemy_direction']
        env.now_step = 0
        
        # print(f"[INFO] 位置设置完成", file=sys.stderr)
        
        # 采集数据的列表
        trajectory = []
        cumulative_reward = 0  # 累计奖励
        
        # 在执行 step 时也重定向 stdout
        sys.stdout = io.StringIO()
        try:
            # 逐步执行action
            for step_idx, action in enumerate(episode_data['actions']):
                # 执行步骤
                obs, reward, done, terminate, trunced, info = env.step(action)
                
                # 更新累计奖励
                cumulative_reward += reward
                
                # 采集数据
                step_data = {
                    'step': step_idx,
                    'action': action,
                    'uuv_x': env.uuv.x,
                    'uuv_y': env.uuv.y,
                    'uuv_z': env.uuv.z,
                    'enemy_x': env.enemy.x,
                    'enemy_y': env.enemy.y,
                    'enemy_z': env.enemy.z,
                    'reward': float(reward),
                    'cumulative_reward': float(cumulative_reward),
                    'reward_details': _convert_reward_details(info.get('total_reward_details', {})),
                    'acoustic_signal': float(env.cumulative_acoustic_signal),
                    'done': done,
                    'terminate': terminate,
                    'trunced': trunced,
                    'result': info.get('result', 'running')
                }
                trajectory.append(step_data)
                
                if done:
                    break
        finally:
            # 恢复 stdout
            sys.stdout = old_stdout
        
        # print(f"[INFO] 回放完成，总共 {len(trajectory)} 步", file=sys.stderr)
        
        # 计算摘要统计
        # summary = _calculate_trajectory_summary(trajectory)
        
        # 输出 JSON 格式的结果
        result = {
            'episode_index': episode_index,
            'episode_info': {
                'uuv_x': episode_data['uuv_x'],
                'uuv_y': episode_data['uuv_y'],
                'uuv_z': episode_data['uuv_z'],
                'enemy_y': episode_data['enemy_y'],
                'enemy_direction': episode_data['enemy_direction'],
                'action_count': episode_data['action_count']
            },
            'trajectory': trajectory,
        #     'summary': summary
        }
        
        # 输出 JSON 到标准输出（只有 JSON）
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # 防止程序后续的清理过程输出内容到 stdout
        sys.stdout = io.StringIO()
        
        return result
        
    except Exception as e:
        # 确保 stdout 已恢复，以便输出错误信息
        sys.stdout = old_stdout
        print(f"[ERROR] 回放过程中出错: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None


def _calculate_trajectory_summary(trajectory: List[dict]) -> dict:
    """计算轨迹摘要统计信息
    
    参数：
        trajectory: 轨迹数据列表
    
    返回：
        包含统计信息的字典
    """
    if not trajectory:
        return {}
    
    # 计算统计信息
    total_reward = sum(d['reward'] for d in trajectory)
    avg_reward = total_reward / len(trajectory)
    total_steps = len(trajectory)
    
    # 查找TL相关统计
    tl_values = [d['tl'] for d in trajectory]
    max_tl = max(tl_values)
    min_tl = min(tl_values)
    avg_tl = sum(tl_values) / len(tl_values)
    
    # 查找最终结果
    final_result = trajectory[-1]['result']
    final_uuv_pos = (trajectory[-1]['uuv_x'], trajectory[-1]['uuv_y'], trajectory[-1]['uuv_z'])
    
    return {
        'total_steps': total_steps,
        'total_reward': float(total_reward),
        'avg_reward_per_step': float(avg_reward),
        'max_tl': float(max_tl),
        'min_tl': float(min_tl),
        'avg_tl': float(avg_tl),
        'final_result': final_result,
        'final_uuv_position': final_uuv_pos,
        'terminated': trajectory[-1]['terminate'],
        'truncated': trajectory[-1]['trunced']
    }





def main():
    """主函数，解析命令行参数并执行相应操作"""
    parser = argparse.ArgumentParser(
        description='强化学习 Step 级日志分析和回放工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法：
  # 统计日志中的游戏轮数
  python step_log_analyzer.py count-games ./output/step_log.txt
  
  # 回放第0轮游戏并采集详细数据
  python step_log_analyzer.py replay-episode ./output/step_log.txt 0
        '''
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='运行模式')
    
    # count-games 模式
    count_parser = subparsers.add_parser('count-games', help='统计日志中的游戏轮数')
    count_parser.add_argument('log_file', type=str, help='Step日志文件路径')
    
    # replay-episode 模式
    replay_parser = subparsers.add_parser('replay-episode', help='回放单轮游戏')
    replay_parser.add_argument('log_file', type=str, help='Step日志文件路径')
    replay_parser.add_argument('episode_index', type=int, help='Episode索引（0-based）')
    replay_parser.add_argument('--config', type=str, default='configs/main_config.yaml', 
                               help='配置文件路径 (默认: configs/main_config.yaml)')
    replay_parser.add_argument('--output-dir', type=str, default=None,
                               help='输出目录路径 (默认: ./output)')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.mode == 'count-games':
            # 统计游戏轮数模式：直接输出整数
        #     print(f"[INFO] 加载日志文件: {args.log_file}", file=sys.stderr)
            analyzer = StepLogAnalyzer(args.log_file)
            total_games = analyzer.count_games()
            
            # 直接输出整数到标准输出
            print(total_games)
        
        elif args.mode == 'replay-episode':
            # 回放单轮游戏模式：输出 JSON 格式结果
        #     print(f"[INFO] 加载日志文件: {args.log_file}", file=sys.stderr)
            analyzer = StepLogAnalyzer(args.log_file)
            
            # 加载配置
        #     print(f"[INFO] 加载配置文件: {args.config}", file=sys.stderr)
            cfg = load_config(args.config)
            
            # 确定输出目录
            output_dir = Path(args.output_dir) if args.output_dir else Path("output")
            
            # 回放episode（输出已在函数中处理）
            replay_episode(analyzer, args.episode_index, cfg, output_dir)
    
    except Exception as e:
        print(f"[ERROR] 程序执行失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
