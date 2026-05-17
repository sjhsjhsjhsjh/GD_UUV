#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习轨迹回放面板后端服务

功能说明：
    - 提供 REST API 接口供前端调用
    - 管理日志文件扫描和轨迹数据获取
    - 集成 step_log_analyzer 的核心功能

API 端点：
    GET  /api/logs                   获取最新日志路径
    POST /api/count-games            计算日志中的游戏轮数
    POST /api/replay-episode         回放指定轮数的游戏
"""

import sys
import json
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from step_log_analyzer import StepLogAnalyzer
from utils.rich_print import print_info, print_error, print_success, print_warn

# ============================================================================
# Flask 应用初始化
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(SCRIPT_DIR), static_url_path='/')
CORS(app)  # 启用 CORS 支持跨域请求

# 应用配置
app.config['JSON_AS_ASCII'] = False  # 支持中文输出
app.config['JSONIFY_MIMETYPE'] = 'application/json; charset=utf-8'

# ============================================================================
# 辅助函数
# ============================================================================


def _get_latest_log_path() -> Optional[Path]:
    """
    扫描 outputs/ 目录，获取最新日期、最新时间的 step_log.txt 路径
    
    返回：
        最新日志文件的 Path 对象，或 None 如果未找到
    
    功能说明：
        遍历 outputs/ 下的日期文件夹（格式 YYYY-MM-DD），选择最新的；
        再在该文件夹中遍历时间文件夹（格式 HH-MM-SS），选择最新的；
        最后返回该文件夹中的 step_log.txt
    """
    outputs_dir = PROJECT_ROOT / 'outputs'
    
    if not outputs_dir.exists():
        print_warn(f"outputs 目录不存在: {outputs_dir}")
        return None
    
    # 获取所有日期文件夹（格式 YYYY-MM-DD）
    date_folders = []
    for item in outputs_dir.iterdir():
        if item.is_dir():
            try:
                # 尝试解析日期格式
                datetime.strptime(item.name, '%Y-%m-%d')
                date_folders.append(item)
            except ValueError:
                continue
    
    if not date_folders:
        print_warn("未找到日期文件夹")
        return None
    
    # 选择最新日期
    latest_date_folder = max(date_folders, key=lambda x: x.name)
    
    # 获取该日期文件夹中所有时间文件夹（格式 HH-MM-SS）
    time_folders = []
    for item in latest_date_folder.iterdir():
        if item.is_dir():
            try:
                # 尝试解析时间格式
                datetime.strptime(item.name, '%H-%M-%S')
                time_folders.append(item)
            except ValueError:
                continue
    
    if not time_folders:
        print_warn(f"未找到时间文件夹: {latest_date_folder}")
        return None
    
    # 选择最新时间
    latest_time_folder = max(time_folders, key=lambda x: x.name)
    
    # 检查 step_log.txt
    log_file = latest_time_folder / 'step_log.txt'
    if not log_file.exists():
        print_warn(f"step_log.txt 不存在: {log_file}")
        return None
    
    return log_file


def _get_all_available_logs() -> List[Dict[str, str]]:
    """
    扫描 outputs/ 目录，获取所有可用的日志文件
    
    返回：
        日志文件列表，每项包含 { "path": "...", "date": "...", "time": "..." }
    
    功能说明：
        按日期和时间倒序排列，供用户手动选择
    """
    outputs_dir = PROJECT_ROOT / 'outputs'
    
    if not outputs_dir.exists():
        return []
    
    logs = []
    
    # 遍历所有日期文件夹
    for date_folder in sorted(outputs_dir.iterdir(), reverse=True):
        if not date_folder.is_dir():
            continue
        
        try:
            datetime.strptime(date_folder.name, '%Y-%m-%d')
        except ValueError:
            continue
        
        # 遍历该日期文件夹中的所有时间文件夹
        for time_folder in sorted(date_folder.iterdir(), reverse=True):
            if not time_folder.is_dir():
                continue
            
            try:
                datetime.strptime(time_folder.name, '%H-%M-%S')
            except ValueError:
                continue
            
            log_file = time_folder / 'step_log.txt'
            if log_file.exists():
                logs.append({
                    'path': str(log_file),
                    'date': date_folder.name,
                    'time': time_folder.name,
                    'display_name': f"{date_folder.name} {time_folder.name}"
                })
    
    return logs


def _validate_log_path(log_path: str) -> tuple[bool, str]:
    """
    验证日志路径有效性
    
    参数：
        log_path: 日志文件路径字符串
    
    返回：
        (是否有效, 错误消息)
    """
    if not log_path:
        return False, "日志路径不能为空"
    
    path = Path(log_path)
    if not path.exists():
        return False, f"日志文件不存在: {log_path}"
    
    if not path.is_file():
        return False, f"路径不是文件: {log_path}"
    
    if path.suffix != '.txt' or path.name != 'step_log.txt':
        return False, f"不是有效的 step_log.txt 文件: {log_path}"
    
    return True, ""


def _load_terrain_data(env) -> dict:
    """
    从环境对象提取地形数据
    
    参数：
        env: Env 环境对象
    
    返回：
        地形数据字典，包含 x_km、y_km、depth_m 等字段
    """
    import numpy as np
    
    try:
        # 加载地形 NPZ 文件
        terrain_npz_path = PROJECT_ROOT / 'output' / 'bty' / 'terrain.npz'
        if not terrain_npz_path.exists():
            print_warn(f"地形文件不存在: {terrain_npz_path}")
            return None
        
        data = np.load(terrain_npz_path)
        bathymetry_2d = data['bathymetry_2d'].astype(np.float32)
        x_coords = data['x_coords'].astype(np.float32)
        y_coords = data['y_coords'].astype(np.float32)
        
        return {
            'x_km': x_coords.tolist(),
            'y_km': y_coords.tolist(),
            'depth_m': bathymetry_2d.tolist(),
            'shape': list(bathymetry_2d.shape)
        }
    
    except Exception as e:
        print_error(f"加载地形数据失败: {e}")
        return None


def _load_tl_data(cfg, enemy_y_km: float) -> dict:
    """
    从配置中加载TL数据
    
    参数：
        cfg: 配置对象
        enemy_y_km: 敌方Y坐标（km）
    
    返回：
        TL数据字典，包含点云坐标和TL值
    """
    import numpy as np
    from omegaconf import OmegaConf
    
    # 调试：确认函数被调用
    (PROJECT_ROOT / 'output' / 'tl_function_called.txt').write_text(f"_load_tl_data called with enemy_y_km={enemy_y_km}\n")
    
    try:
        # 从配置中获取TL文件路径
        tl_file = PROJECT_ROOT / 'TLdata' / 'average_TL_results.txt'
        
        if not tl_file.exists():
            print_warn(f"TL文件不存在: {tl_file}")
            return None
        
        # 读取TL数据
        rows = []
        for line in tl_file.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                values = [float(x) for x in line.replace(',', ' ').split()]
                if len(values) >= 5:
                    rows.append(values[:5])
            except ValueError:
                continue
        
        if not rows:
            print_warn(f"TL文件没有有效数据: {tl_file}")
            return None
        
        rows = np.array(rows, dtype=np.float32)
        
        # 获取采样步长
        try:
            sampling_x_step = float(cfg.env.sampling_x_step) if hasattr(cfg, 'env') else 100
            sampling_y_step = float(cfg.env.sampling_y_step) if hasattr(cfg, 'env') else 100
            sampling_z_step = float(cfg.env.sampling_z_step) if hasattr(cfg, 'env') else 50
        except:
            sampling_x_step = 100
            sampling_y_step = 100
            sampling_z_step = 50
        
        tl_color_cap_db = 120
        
        # 转换网格坐标为实际距离
        # 列格式: enemy_y_grid, uuv_x_grid, uuv_y_grid, uuv_z_grid, tl_db
        enemy_y_km_vals = rows[:, 0] * sampling_y_step / 1000.0
        uuv_x_km_vals = rows[:, 1] * sampling_x_step / 1000.0
        uuv_y_km_vals = rows[:, 2] * sampling_y_step / 1000.0
        uuv_z_m_vals = rows[:, 3] * sampling_z_step
        tl_vals = rows[:, 4]
        
        # 将0 dB替换为限幅值
        tl_vals = np.where(tl_vals == 0, tl_color_cap_db, tl_vals)
        
        # 筛选与当前敌方Y坐标相关的数据（容差±0.5km）
        mask = np.abs(enemy_y_km_vals - enemy_y_km) <= 0.5
        
        return {
            'x_km': uuv_x_km_vals[mask].tolist(),
            'y_km': uuv_y_km_vals[mask].tolist(),
            'z_m': uuv_z_m_vals[mask].tolist(),
            'tl_db': tl_vals[mask].tolist(),
            'stats': {
                'valid_count': int(np.sum(mask)),
                'tl_min_db': float(np.min(tl_vals[mask])) if np.sum(mask) > 0 else 0,
                'tl_max_db': float(np.max(tl_vals[mask])) if np.sum(mask) > 0 else 0,
                'tl_mean_db': float(np.mean(tl_vals[mask])) if np.sum(mask) > 0 else 0,
            }
        }
    
    except Exception as e:
        print_error(f"加载TL数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# API 端点
# ============================================================================


@app.route('/api/logs', methods=['GET'])
def get_logs():
    """
    获取日志文件列表
    
    返回格式：
        {
            "success": true,
            "latest": {
                "path": "/path/to/step_log.txt",
                "date": "2026-05-13",
                "time": "19-08-10"
            },
            "all": [
                { "path": "...", "date": "...", "time": "...", "display_name": "..." },
                ...
            ]
        }
    """
    try:
        latest = _get_latest_log_path()
        all_logs = _get_all_available_logs()
        
        result = {
            'success': True,
            'latest': {
                'path': str(latest),
                'date': latest.parent.name if latest else None,
                'time': latest.parent.parent.name if latest else None
            } if latest else None,
            'all': all_logs
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        print_error(f"获取日志列表失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/count-games', methods=['POST'])
def count_games():
    """
    计算日志中的游戏轮数
    
    请求格式：
        {
            "log_path": "/path/to/step_log.txt"
        }
    
    返回格式：
        {
            "success": true,
            "count": 42,
            "log_path": "/path/to/step_log.txt"
        }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': '请求体为空'
            }), 400
        
        log_path = data.get('log_path', '').strip()
        
        # 验证路径
        valid, error_msg = _validate_log_path(log_path)
        if not valid:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # 创建分析器并计算游戏数
        analyzer = StepLogAnalyzer(log_path)
        count = analyzer.count_games()
        
        print_success(f"统计游戏数: {count} (日志: {log_path})")
        
        return jsonify({
            'success': True,
            'count': count,
            'log_path': log_path
        }), 200
    
    except Exception as e:
        print_error(f"计算游戏数失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/replay-episode', methods=['POST'])
def replay_episode_api():
    """
    回放单个游戏轮次
    
    请求格式：
        {
            "log_path": "/path/to/step_log.txt",
            "episode_index": 0
        }
    
    返回格式：
        {
            "success": true,
            "trajectory": {
                "episode_index": 0,
                "episode_info": { ... },
                "trajectory": [ { ... }, ... ]
            }
        }
    """
    try:
        # 调试：记录函数入口
        (PROJECT_ROOT / 'output' / 'replay_api_start.txt').write_text("replay_episode_api called\n")
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': '请求体为空'
            }), 400
        
        log_path = data.get('log_path', '').strip()
        episode_index = data.get('episode_index')
        
        # 验证路径
        valid, error_msg = _validate_log_path(log_path)
        if not valid:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # 验证 episode_index
        if episode_index is None:
            return jsonify({
                'success': False,
                'error': 'episode_index 不能为空'
            }), 400
        
        try:
            episode_index = int(episode_index)
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': f'episode_index 必须是整数: {episode_index}'
            }), 400
        
        if episode_index < 0:
            return jsonify({
                'success': False,
                'error': f'episode_index 不能为负数: {episode_index}'
            }), 400
        
        # 创建分析器
        analyzer = StepLogAnalyzer(log_path)
        
        # 验证 episode_index 是否在有效范围内
        total_games = analyzer.count_games()
        if episode_index >= total_games:
            return jsonify({
                'success': False,
                'error': f'episode_index 越界: {episode_index} >= {total_games}'
            }), 400
        
        # 导入必要的模块
        from step_log_analyzer import load_config, _convert_reward_details
        from env.env import Env
        import os
        import io
        
        # 加载配置
        cfg = load_config(str(PROJECT_ROOT / 'configs' / 'main_config.yaml'))
        
        # 获取 episode 信息
        episode_data = analyzer.get_episode(episode_index)
        
        # 改变工作目录和重定向 stdout 来初始化 Env（Env 初始化时会产生大量输出和 matplotlib 警告）
        old_cwd = os.getcwd()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            # 改变工作目录到项目根目录
            os.chdir(PROJECT_ROOT)
            
            # 重定向 stdout 和 stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            
            try:
                # 创建环境实例
                env = Env(cfg, output_dir=PROJECT_ROOT / 'output', enable_step_log_server=False)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            
            # 重置环境
            env.reset()
            
            # 覆盖初始位置为日志中记录的位置
            env.uuv.x = episode_data['uuv_x']
            env.uuv.y = episode_data['uuv_y']
            env.uuv.z = episode_data['uuv_z']
            env.enemy.y = episode_data['enemy_y']
            env.enemy_forward_direction = episode_data['enemy_direction']
            env.now_step = 0
            
            # 采集数据的列表
            trajectory = []
            cumulative_reward = 0  # 累计奖励
            
            # 在执行 step 时也重定向 stdout
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
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
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            
            # 构建返回结果
            print_info("开始加载地形和TL数据...")
            terrain_data = _load_terrain_data(env)
            print_info(f"地形数据: {type(terrain_data)}, {terrain_data is not None}")
            
            # 将日志中的敌方Y网格坐标转换为km
            sampling_y_step = float(cfg.env.sampling_y_step) if hasattr(cfg, 'env') else 100
            enemy_y_km = episode_data['enemy_y'] * sampling_y_step / 1000.0
            print_info(f"敌方Y: {episode_data['enemy_y']} (grid) -> {enemy_y_km:.3f} (km)")
            
            # 调试：在调用_load_tl_data前写入文件
            debug_path = PROJECT_ROOT / 'output' / 'before_tl_call.txt'
            debug_path.write_text(f"About to call _load_tl_data with enemy_y_km={enemy_y_km}\n")
            
            tl_data = _load_tl_data(cfg, enemy_y_km)
            
            # 调试：在调用_load_tl_data后写入文件
            debug_path2 = PROJECT_ROOT / 'output' / 'after_tl_call.txt'
            debug_path2.write_text(f"After _load_tl_data: tl_data is None: {tl_data is None}\n")
            
            print_info(f"TL数据: {type(tl_data)}, {tl_data is not None}")
            if tl_data and 'stats' in tl_data:
                print_info(f"TL统计: {tl_data['stats']}")
            
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
                'terrain': terrain_data,
                'tl': tl_data
            }
            
            print_success(f"回放完成: episode {episode_index}, {len(trajectory)} 步")
            
            return jsonify({
                'success': True,
                'trajectory': result
            }), 200
        
        finally:
            os.chdir(old_cwd)
    
    except Exception as e:
        print_error(f"回放游戏失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    健康检查端点
    
    返回格式：
        {
            "status": "ok",
            "version": "1.0.0"
        }
    """
    return jsonify({
        'status': 'ok',
        'version': '1.0.0'
    }), 200


@app.route('/', methods=['GET'])
def index():
    """
    主页路由，返回 index.html
    """
    return send_from_directory(Path(__file__).parent, 'index.html')


@app.route('/<path:path>', methods=['GET'])
def serve_static(path):
    """
    提供静态文件（CSS、JS、等）
    """
    return send_from_directory(Path(__file__).parent, path)


# ============================================================================
# 错误处理
# ============================================================================


@app.errorhandler(404)
def not_found(error):
    """404 错误处理"""
    return jsonify({
        'success': False,
        'error': '端点不存在'
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """405 方法不允许"""
    return jsonify({
        'success': False,
        'error': '方法不允许'
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """500 内部服务器错误"""
    return jsonify({
        'success': False,
        'error': '内部服务器错误'
    }), 500


# ============================================================================
# 应用入口
# ============================================================================


if __name__ == '__main__':
    print_info("="*70)
    print_info("强化学习轨迹回放面板 - 后端服务")
    print_info("="*70)
    print_info(f"项目根目录: {PROJECT_ROOT}")
    print_info(f"Flask 应用启动于: http://127.0.0.1:5000")
    print_info("按 Ctrl+C 停止服务")
    print_info("="*70)
    
    # 以调试模式启动（开发环境）
    # 生产环境应使用 gunicorn 或类似 WSGI 服务器
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,
        use_reloader=True
    )
