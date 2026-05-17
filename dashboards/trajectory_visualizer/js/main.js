/**
 * 强化学习轨迹回放面板 - 前端主逻辑
 * 
 * 功能说明：
 *   1. 日志文件管理 - 自动加载最新日志或手动选择
 *   2. 游戏轮数统计 - 调用后端 count-games API
 *   3. 轨迹数据获取 - 调用后端 replay-episode API
 *   4. 3D 轨迹绘制 - 使用 Plotly 绘制 UUV 和敌方轨迹
 *   5. 交互展示 - 鼠标悬停显示 step 详情
 */

// ============================================================================
// 全局状态管理
// ============================================================================

const state = {
  // 日志和数据相关
  logPath: null,              // 当前日志文件路径
  totalGames: 0,              // 日志中的总游戏轮数
  currentEpisodeIndex: null,  // 当前加载的 episode 索引
  trajectoryData: null,       // 从后端获取的完整轨迹 JSON
  
  // 轨迹数据提取结果
  uuvTrajectory: [],          // UUV 坐标序列：[{x, y, z, step, ...}, ...]
  enemyTrajectory: [],        // 敌方坐标序列：[{enemy_x, enemy_y, enemy_z, step, ...}, ...]
  stepDetails: [],            // 每个 step 的完整详细信息
  
  // 地形和TL数据
  terrain: null,              // 地形数据：{x_km, y_km, depth_m, shape}
  tl: null,                   // TL点云数据：{x_km, y_km, z_m, tl_db, stats}
  
  // 显示控制
  showTerrain: true,          // 是否显示地形
  showTl: true,               // 是否显示TL点云
  tlColorCapDb: 120,          // TL颜色限幅值（dB）
  
  // UI 状态
  selectedStep: null,         // 当前选中的 step（鼠标悬停）
  apiBaseUrl: 'http://127.0.0.1:5000/api',
};

// ============================================================================
// UI 元素引用
// ============================================================================

const ui = {
  // 日志选择
  logPath: document.getElementById('logPath'),
  browseLogs: document.getElementById('browseLogs'),
  refreshLogs: document.getElementById('refreshLogs'),
  
  // 游戏选择
  gameCount: document.getElementById('gameCount'),
  episodeInput: document.getElementById('episodeInput'),
  loadEpisode: document.getElementById('loadEpisode'),
  
  // 游戏信息
  episodeInfo: document.getElementById('episodeInfo'),
  
  // Step 详情
  stepDetails: document.getElementById('stepDetails'),
  
  // 状态文本和 Plotly 容器
  statusText: document.getElementById('statusText'),
  plot: document.getElementById('plot'),
};

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 更新状态文本
 */
function setStatus(message, type = 'info') {
  const icons = {
    'info': 'ℹ️',
    'success': '✓',
    'error': '✗',
    'warn': '⚠️'
  };
  ui.statusText.textContent = `${icons[type] || '•'} ${message}`;
  console.log(`[${type.toUpperCase()}] ${message}`);
}

/**
 * 显示错误提示
 */
function showError(message) {
  setStatus(message, 'error');
  alert(`❌ 错误：${message}`);
}

/**
 * 格式化数字为指定小数位
 */
function formatNumber(value, decimals = 2) {
  if (value === null || value === undefined) return '-';
  return Number(value).toFixed(decimals);
}

/**
 * 坐标转换：原始坐标 -> 实际米数 -> km/m
 * 转换规则：
 *   - 横纵坐标 * 100 = 实际米数
 *   - 深度坐标 * 50 = 实际米数
 *   - 显示时转换为 km（xy）或 m（z）
 * @param {number} rawCoord - 原始坐标值
 * @param {string} axis - 轴类型：'xy' 或 'z'
 * @returns {number} 转换后的坐标（km 或 m）
 */
function convertCoordinate(rawCoord, axis = 'xy') {
  if (axis === 'xy') {
    return rawCoord * 100 / 1000;  // 原始值 * 100 (米) / 1000 = km
  } else {
    return rawCoord * 50;  // 原始值 * 50 = 米
  }
}

/**
 * 构建详情面板的 HTML
 */
function buildStepDetailsHTML(stepData) {
  if (!stepData) {
    return '<div class="detail-empty">鼠标悬停轨迹点以显示详情</div>';
  }

  const rewardDetails = stepData.reward_details || {};
  
  let html = `
    <div class="detail-item">
      <div class="detail-label">Step ${stepData.step} / Action ${stepData.action}</div>
      <div class="detail-value">${stepData.result || 'running'}</div>
    </div>

    <div class="detail-item">
      <div class="detail-label">UUV 位置</div>
      <div class="detail-row">
        <div class="detail-row-left">
          <div class="detail-row-value">
            (${formatNumber(stepData.uuv_x_km, 3)}, ${formatNumber(stepData.uuv_y_km, 3)}, ${formatNumber(stepData.uuv_z_m, 1)})<br/>
            <span style="font-size: 0.85em; color: #9ca3af;">km, km, m</span>
          </div>
        </div>
      </div>
    </div>

    <div class="detail-item">
      <div class="detail-label">敌方位置</div>
      <div class="detail-row">
        <div class="detail-row-left">
          <div class="detail-row-value">
            (${formatNumber(stepData.enemy_x_km, 3)}, ${formatNumber(stepData.enemy_y_km, 3)}, ${formatNumber(stepData.enemy_z_m, 1)})<br/>
            <span style="font-size: 0.85em; color: #9ca3af;">km, km, m</span>
          </div>
        </div>
      </div>
    </div>

    <div class="reward-section">
      <div class="detail-label">即时奖励</div>
      <div class="reward-item">
        <span class="reward-name">Reward</span>
        <span class="reward-value ${stepData.reward >= 0 ? 'reward-positive' : 'reward-negative'}">
          ${formatNumber(stepData.reward, 4)}
        </span>
      </div>
      
      <div class="reward-item">
        <span class="reward-name">Cumulative</span>
        <span class="reward-value ${stepData.cumulative_reward >= 0 ? 'reward-positive' : 'reward-negative'}">
          ${formatNumber(stepData.cumulative_reward, 2)}
        </span>
      </div>

      ${rewardDetails['sum_stealth_reward'] !== undefined ? `
        <div class="reward-item">
          <span class="reward-name">Stealth</span>
          <span class="reward-value ${rewardDetails['sum_stealth_reward'] >= 0 ? 'reward-positive' : 'reward-negative'}">
            ${formatNumber(rewardDetails['sum_stealth_reward'], 4)}
          </span>
        </div>
      ` : ''}

      ${rewardDetails['sum_approach_reward'] !== undefined ? `
        <div class="reward-item">
          <span class="reward-name">Approach</span>
          <span class="reward-value ${rewardDetails['sum_approach_reward'] >= 0 ? 'reward-positive' : 'reward-negative'}">
            ${formatNumber(rewardDetails['sum_approach_reward'], 4)}
          </span>
        </div>
      ` : ''}

      ${rewardDetails['sum_tl_gradient_reward'] !== undefined ? `
        <div class="reward-item">
          <span class="reward-name">TL Gradient</span>
          <span class="reward-value ${rewardDetails['sum_tl_gradient_reward'] >= 0 ? 'reward-positive' : 'reward-negative'}">
            ${formatNumber(rewardDetails['sum_tl_gradient_reward'], 4)}
          </span>
        </div>
      ` : ''}

      ${rewardDetails['sum_area_average_tl_reward'] !== undefined ? `
        <div class="reward-item">
          <span class="reward-name">Area TL</span>
          <span class="reward-value ${rewardDetails['sum_area_average_tl_reward'] >= 0 ? 'reward-positive' : 'reward-negative'}">
            ${formatNumber(rewardDetails['sum_area_average_tl_reward'], 4)}
          </span>
        </div>
      ` : ''}

      ${rewardDetails['sum_fixed_time_penalty'] !== undefined ? `
        <div class="reward-item">
          <span class="reward-name">Time Penalty</span>
          <span class="reward-value ${rewardDetails['sum_fixed_time_penalty'] >= 0 ? 'reward-positive' : 'reward-negative'}">
            ${formatNumber(rewardDetails['sum_fixed_time_penalty'], 4)}
          </span>
        </div>
      ` : ''}
    </div>

    <div class="detail-item">
      <div class="detail-label">声学信号</div>
      <div class="detail-value">${formatNumber(stepData.acoustic_signal, 2)}</div>
    </div>
  `;

  return html;
}

// ============================================================================
// 日志管理模块
// ============================================================================

/**
 * 加载最新日志
 */
async function loadLatestLog() {
  try {
    setStatus('正在加载日志列表...', 'info');
    
    const response = await fetch(`${state.apiBaseUrl}/logs`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    
    const data = await response.json();
    if (!data.success) throw new Error(data.error);
    
    if (!data.latest || !data.latest.path) {
      showError('未找到最新日志文件');
      return false;
    }
    
    state.logPath = data.latest.path;
    ui.logPath.value = data.latest.path;
    
    setStatus('✓ 日志加载完成', 'success');
    return true;
    
  } catch (error) {
    showError(`加载日志失败：${error.message}`);
    return false;
  }
}

/**
 * 刷新游戏轮数
 */
async function refreshGameCount() {
  try {
    if (!state.logPath) {
      showError('未选择日志文件');
      return false;
    }
    
    setStatus('正在统计游戏轮数...', 'info');
    
    const response = await fetch(`${state.apiBaseUrl}/count-games`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ log_path: state.logPath })
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    
    const data = await response.json();
    if (!data.success) throw new Error(data.error);
    
    state.totalGames = data.count;
    ui.gameCount.textContent = data.count;
    
    setStatus(`✓ 共 ${data.count} 轮游戏`, 'success');
    return true;
    
  } catch (error) {
    showError(`统计游戏轮数失败：${error.message}`);
    return false;
  }
}

/**
 * 加载指定轮的游戏
 */
async function loadEpisode(episodeIndex) {
  try {
    if (!state.logPath) {
      showError('未选择日志文件');
      return false;
    }
    
    if (episodeIndex < 0 || episodeIndex >= state.totalGames) {
      showError(`轮数越界，有效范围：0-${state.totalGames - 1}`);
      return false;
    }
    
    setStatus(`正在回放第 ${episodeIndex} 轮游戏...`, 'info');
    
    const response = await fetch(`${state.apiBaseUrl}/replay-episode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        log_path: state.logPath, 
        episode_index: episodeIndex 
      })
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    
    const data = await response.json();
    if (!data.success) throw new Error(data.error);
    
    state.trajectoryData = data.trajectory;
    state.currentEpisodeIndex = episodeIndex;
    
    // 解析和处理轨迹数据
    parseTrajectoryData(data.trajectory);
    
    // 更新 UI
    updateEpisodeInfo(data.trajectory);
    
    // 绘制轨迹
    drawTrajectory();
    
    setStatus(`✓ 回放完成：${state.stepDetails.length} 步`, 'success');
    return true;
    
  } catch (error) {
    showError(`回放游戏失败：${error.message}`);
    return false;
  }
}

// ============================================================================
// 数据解析模块
// ============================================================================

/**
 * 解析轨迹数据
 * 转换坐标：原始值 -> km/m
 */
function parseTrajectoryData(trajectoryData) {
  const trajectory = trajectoryData.trajectory || [];
  const episodeInfo = trajectoryData.episode_info || {};
  
  state.uuvTrajectory = [];
  state.enemyTrajectory = [];
  state.stepDetails = [];
  
  // 提取地形和TL数据
  state.terrain = trajectoryData.terrain || null;
  state.tl = trajectoryData.tl || null;
  
  for (const step of trajectory) {
    // 转换坐标
    const uuv_x_km = convertCoordinate(step.uuv_x, 'xy');
    const uuv_y_km = convertCoordinate(step.uuv_y, 'xy');
    const uuv_z_m = convertCoordinate(step.uuv_z, 'z');
    const enemy_x_km = convertCoordinate(step.enemy_x, 'xy');
    const enemy_y_km = convertCoordinate(step.enemy_y, 'xy');
    const enemy_z_m = convertCoordinate(step.enemy_z, 'z');
    
    // 提取 UUV 轨迹（使用转换后的坐标）
    state.uuvTrajectory.push({
      x: uuv_x_km,
      y: uuv_y_km,
      z: uuv_z_m,
      step: step.step,
      action: step.action
    });
    
    // 提取敌方轨迹（使用转换后的坐标）
    state.enemyTrajectory.push({
      x: enemy_x_km,
      y: enemy_y_km,
      z: enemy_z_m,
      step: step.step
    });
    
    // 保存详细信息（添加转换后的坐标字段）
    const stepWithConvertedCoords = {
      ...step,
      uuv_x_km,
      uuv_y_km,
      uuv_z_m,
      enemy_x_km,
      enemy_y_km,
      enemy_z_m
    };
    state.stepDetails.push(stepWithConvertedCoords);
  }
}

/**
 * 更新游戏信息显示
 */
function updateEpisodeInfo(trajectoryData) {
  const info = trajectoryData.episode_info || {};
  const trajectory = trajectoryData.trajectory || [];
  
  const lastStep = trajectory.length > 0 ? trajectory[trajectory.length - 1] : null;
  const finalResult = lastStep ? lastStep.result : '未完成';
  const totalSteps = trajectory.length;
  const finalReward = lastStep ? lastStep.cumulative_reward : 0;
  
  // 转换初始坐标
  const init_uuv_x_km = formatNumber(convertCoordinate(info.uuv_x, 'xy'), 3);
  const init_uuv_y_km = formatNumber(convertCoordinate(info.uuv_y, 'xy'), 3);
  const init_uuv_z_m = formatNumber(convertCoordinate(info.uuv_z, 'z'), 1);
  const init_enemy_y_km = formatNumber(convertCoordinate(info.enemy_y, 'xy'), 3);
  
  ui.episodeInfo.innerHTML = `
    <div class="meta-item">
      <div class="meta-label">初始 UUV 位置</div>
      <div class="meta-value">(${init_uuv_x_km}, ${init_uuv_y_km}, ${init_uuv_z_m})<br/>
      <span style="font-size: 0.85em; color: #9ca3af;">km, km, m</span></div>
    </div>
    <div class="meta-item">
      <div class="meta-label">初始敌方 Y</div>
      <div class="meta-value">${init_enemy_y_km} km</div>
    </div>
    <div class="meta-item">
      <div class="meta-label">总 Action 数</div>
      <div class="meta-value">${info.action_count}</div>
    </div>
    <div class="meta-item">
      <div class="meta-label">实际 Step 数</div>
      <div class="meta-value">${totalSteps}</div>
    </div>
    <div class="meta-item">
      <div class="meta-label">最终结果</div>
      <div class="meta-value">${finalResult}</div>
    </div>
    <div class="meta-item">
      <div class="meta-label">最终奖励</div>
      <div class="meta-value">${formatNumber(finalReward, 2)}</div>
    </div>
  `;
}

// ============================================================================
// Plotly 轨迹绘制模块
// ============================================================================

/**
 * 生成 UUV 轨迹 trace
 */
function toUUVTrajectoryTrace() {
  return {
    type: 'scatter3d',
    mode: 'lines+markers',
    name: 'UUV 轨迹',
    x: state.uuvTrajectory.map(p => p.x),
    y: state.uuvTrajectory.map(p => p.y),
    z: state.uuvTrajectory.map(p => -p.z),  // 深度为负数
    line: {
      color: '#4cc9f0',  // 亮青色
      width: 3
    },
    marker: {
      size: 4,
      color: '#4cc9f0',
      opacity: 0.9,
      line: { color: '#ffffff', width: 0.5 }
    },
    text: state.uuvTrajectory.map((p, i) => `Step ${p.step}: (${p.x.toFixed(3)}km, ${p.y.toFixed(3)}km, ${p.z.toFixed(1)}m)`),
    hovertemplate: '%{text}<extra></extra>',
    visible: true
  };
}

/**
 * 生成敌方轨迹 trace
 */
function toEnemyTrajectoryTrace() {
  return {
    type: 'scatter3d',
    mode: 'lines+markers',
    name: '敌方轨迹',
    x: state.enemyTrajectory.map(p => p.x),
    y: state.enemyTrajectory.map(p => p.y),
    z: state.enemyTrajectory.map(p => -p.z),  // 深度为负数
    line: {
      color: '#ff6b6b',  // 红色
      width: 3,
      dash: 'dash'
    },
    marker: {
      size: 5,
      color: '#ff6b6b',
      opacity: 0.9,
      symbol: 'diamond',
      line: { color: '#ffffff', width: 0.5 }
    },
    text: state.enemyTrajectory.map((p, i) => `Enemy: (${p.x.toFixed(3)}km, ${p.y.toFixed(3)}km, ${p.z.toFixed(1)}m)`),
    hovertemplate: '%{text}<extra></extra>',
    visible: true
  };
}

/**
 * 生成关键点标记 trace（起点、终点）
 */
function toKeyPointsTrace() {
  const points = [];
  const labels = [];
  const colors = [];
  
  if (state.uuvTrajectory.length > 0) {
    // 起点
    const start = state.uuvTrajectory[0];
    points.push([start.x, start.y, -start.z]);
    labels.push(`起点 (${start.x.toFixed(3)}km, ${start.y.toFixed(3)}km, ${start.z.toFixed(1)}m)`);
    colors.push('#4ade80');  // 绿色
    
    // 终点
    const end = state.uuvTrajectory[state.uuvTrajectory.length - 1];
    points.push([end.x, end.y, -end.z]);
    labels.push(`终点 (${end.x.toFixed(3)}km, ${end.y.toFixed(3)}km, ${end.z.toFixed(1)}m)`);
    colors.push('#f87171');  // 红色
  }
  
  if (points.length === 0) return null;
  
  return {
    type: 'scatter3d',
    mode: 'markers',
    name: '关键点',
    x: points.map(p => p[0]),
    y: points.map(p => p[1]),
    z: points.map(p => p[2]),
    marker: {
      size: 8,
      color: colors,
      symbol: 'star',
      line: { color: '#ffffff', width: 1 }
    },
    text: labels,
    hovertemplate: '%{text}<extra></extra>',
    visible: true
  };
}

/**
 * 生成地形 trace
 */
function toTerrainTrace() {
  if (!state.terrain) return null;
  
  const terrain = state.terrain;
  const xAxis = terrain.x_km;
  const yAxis = terrain.y_km;
  const depth = terrain.depth_m;
  
  return {
    type: 'surface',
    name: '三维地形',
    visible: state.showTerrain,
    x: xAxis.map((v) => Number(v)),
    y: yAxis.map((v) => Number(v)),
    z: depth.map((row) => row.map((depthValue) => -Number(depthValue))),
    surfacecolor: depth.map((row) => row.map((depthValue) => Number(depthValue))),
    colorscale: 'Cividis',
    opacity: 0.42,
    showscale: false,
    hovertemplate: '地形面<br>x=%{x:.3f}km<br>y=%{y:.3f}km<br>z=%{z:.2f}m<extra></extra>',
  };
}

/**
 * 生成 TL 点云 trace
 */
function toTlTrace() {
  if (!state.tl || state.tl.x_km.length === 0) return null;
  
  const tlData = state.tl;
  const tlCapDb = state.tlColorCapDb;
  const stats = tlData.stats || {};
  const cmin = stats.tl_min_db || 0;
  const cmax = stats.tl_max_db || tlCapDb;
  
  return {
    type: 'scatter3d',
    mode: 'markers',
    name: '传播损失点云',
    visible: state.showTl,
    x: tlData.x_km.map(v => Number(v)),
    y: tlData.y_km.map(v => Number(v)),
    z: tlData.z_m.map(v => -Number(v)),
    marker: {
      size: 4.0,
      opacity: 0.78,
      color: tlData.tl_db.map(v => Math.min(Number(v), tlCapDb)),
      colorscale: 'Plasma',
      cmin: cmin,
      cmax: cmax,
      colorbar: {
        title: 'TL (dB)',
        thickness: 14,
        len: 0.75,
      },
    },
    hovertemplate: 'TL点<br>x=%{x:.3f}km<br>y=%{y:.3f}km<br>z=%{z:.2f}m<br>TL=%{marker.color:.3f}dB<extra></extra>',
  };
}

/**
 * 绘制轨迹
 */
function drawTrajectory() {
  const traces = [];
  
  // 添加地形（背景）
  const terrainTrace = toTerrainTrace();
  if (terrainTrace) {
    traces.push(terrainTrace);
  }
  
  // 添加TL点云（背景）
  const tlTrace = toTlTrace();
  if (tlTrace) {
    traces.push(tlTrace);
  }
  
  // 添加 UUV 轨迹
  traces.push(toUUVTrajectoryTrace());
  
  // 添加敌方轨迹
  traces.push(toEnemyTrajectoryTrace());
  
  // 添加关键点
  const keyPointsTrace = toKeyPointsTrace();
  if (keyPointsTrace) {
    traces.push(keyPointsTrace);
  }
  
  // 布局设置
  const layout = {
    title: {
      text: `第 ${state.currentEpisodeIndex} 轮游戏轨迹（${state.stepDetails.length} 步）`,
      font: { size: 16, color: '#e5eefb' }
    },
    scene: {
      xaxis: { title: 'X (km)', backgroundcolor: 'rgba(0,0,0,0.1)' },
      yaxis: { title: 'Y (km)', backgroundcolor: 'rgba(0,0,0,0.1)' },
      zaxis: { title: 'Z (m, 海平面以下为负)', backgroundcolor: 'rgba(0,0,0,0.1)' },
      camera: {
        eye: { x: 1.35, y: 1.35, z: 0.95 }
      }
    },
    hovermode: 'closest',
    margin: { l: 0, r: 0, b: 0, t: 40 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0.1)',
    font: { color: '#e5eefb' },
    showlegend: true,
    legend: {
      x: 0.02,
      y: 0.98,
      bgcolor: 'rgba(10, 23, 41, 0.8)',
      bordercolor: 'rgba(118, 155, 200, 0.2)',
      borderwidth: 1
    }
  };
  
  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false
  };
  
  Plotly.newPlot(ui.plot, traces, layout, config);
  
  // 绑定 hover 事件
  ui.plot.on('plotly_hover', function(data) {
    if (data.points.length > 0) {
      const point = data.points[0];
      
      // 判断是哪条轨迹
      if (point.curveNumber === 0) {
        // UUV 轨迹
        const stepIndex = point.pointNumber;
        if (stepIndex >= 0 && stepIndex < state.stepDetails.length) {
          state.selectedStep = state.stepDetails[stepIndex];
          updateStepDetails();
        }
      }
    }
  });
  
  // 注意：移除 plotly_unhover 事件处理，保留 step 详情显示
}

/**
 * 更新 Step 详情面板
 */
function updateStepDetails() {
  ui.stepDetails.innerHTML = buildStepDetailsHTML(state.selectedStep);
}

// ============================================================================
// 事件处理
// ============================================================================

/**
 * 初始化事件监听
 */
function initEventListeners() {
  // 刷新日志
  ui.refreshLogs.addEventListener('click', async () => {
    const success = await refreshGameCount();
    if (success) {
      ui.episodeInput.focus();
    }
  });
  
  // 加载游戏
  ui.loadEpisode.addEventListener('click', async () => {
    const index = parseInt(ui.episodeInput.value);
    if (isNaN(index)) {
      showError('请输入有效的轮数');
      return;
    }
    await loadEpisode(index);
  });
  
  // 回车加载
  ui.episodeInput.addEventListener('keypress', async (e) => {
    if (e.key === 'Enter') {
      const index = parseInt(ui.episodeInput.value);
      if (!isNaN(index)) {
        await loadEpisode(index);
      }
    }
  });
  
  // 浏览日志（占位，可在后续实现选择对话框）
  ui.browseLogs.addEventListener('click', () => {
    setStatus('选择日志功能待实现', 'info');
  });
}

// ============================================================================
// 初始化
// ============================================================================

/**
 * 页面初始化
 */
async function init() {
  try {
    setStatus('系统初始化中...', 'info');
    
    // 绑定事件
    initEventListeners();
    
    // 加载最新日志
    const logLoaded = await loadLatestLog();
    if (!logLoaded) return;
    
    // 自动统计游戏轮数
    await refreshGameCount();
    
    setStatus('✓ 系统就绪', 'success');
    
  } catch (error) {
    showError(`初始化失败：${error.message}`);
  }
}

// ============================================================================
// 启动
// ============================================================================

// 页面加载完成后初始化
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
