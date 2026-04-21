const state = {
  payload: null,
  runtimeConfig: null,
  currentEnemyIndex: 0,
  showTerrain: true,
  showPoints: true,
  pointSize: 3.2,
  cameraCenter: {
    x: 0,
    y: 0,
    z: 0,
  },
};

const ui = {
  enemySelect: document.getElementById('enemySelect'),
  reloadBtn: document.getElementById('reloadBtn'),
  runBadge: document.getElementById('runBadge'),
  runMeta: document.getElementById('runMeta'),
  enemyList: document.getElementById('enemyList'),
  enemyMeta: document.getElementById('enemyMeta'),
  toggleTerrain: document.getElementById('toggleTerrain'),
  togglePoints: document.getElementById('togglePoints'),
  pointSizeInput: document.getElementById('pointSizeInput'),
  statusText: document.getElementById('statusText'),
  plot: document.getElementById('plot'),
};

function setStatus(text) {
  ui.statusText.textContent = text;
}

function appendMeta(container, label, value) {
  const card = document.createElement('div');
  card.className = 'meta-item';
  card.innerHTML = `<div class="meta-label">${label}</div><div class="meta-value">${value}</div>`;
  container.appendChild(card);
}

function formatValue(value, digits = 3) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : '-';
}

function clamp(value, minValue, maxValue) {
  return Math.min(maxValue, Math.max(minValue, value));
}

function toTerrainTrace(payload, runtimeConfig) {
  const terrain = payload.terrain;
  return {
    type: 'surface',
    name: '三维地形',
    visible: state.showTerrain,
    x: terrain.x_km.map((v) => Number(v)),
    y: terrain.y_km.map((v) => Number(v)),
    z: terrain.depth_m.map((row) => row.map((depthValue) => -Number(depthValue))),
    surfacecolor: terrain.depth_m.map((row) => row.map((depthValue) => Number(depthValue))),
    colorscale: 'Cividis',
    opacity: Number(runtimeConfig.plotly_config.terrain_opacity || 0.42),
    showscale: false,
    hovertemplate: '地形面<br>x=%{x:.3f}km<br>y=%{y:.3f}km<br>z=%{z:.2f}m<extra></extra>',
  };
}

function getCurrentEnemy() {
  return state.payload.enemy_positions[state.currentEnemyIndex];
}

function toKeyPointTrace() {
  const payload = state.payload;
  const runtimeConfig = state.runtimeConfig;
  const colorMin = Number.isFinite(Number(runtimeConfig.color_min))
    ? Number(runtimeConfig.color_min)
    : Number(payload.stats.color_min_auto);
  const colorMax = Number.isFinite(Number(runtimeConfig.color_max))
    ? Number(runtimeConfig.color_max)
    : Number(payload.stats.color_max_auto);

  return {
    type: 'scatter3d',
    mode: 'markers',
    name: '关键点点云',
    visible: state.showPoints,
    x: payload.points.x_km.map((value) => Number(value)),
    y: payload.points.y_km.map((value) => Number(value)),
    z: payload.points.z_plot_m.map((value) => Number(value)),
    marker: {
      size: Number(state.pointSize),
      opacity: Number(runtimeConfig.plotly_config.marker_opacity || 0.82),
      color: payload.points.color_value.map((value) => Number(value)),
      colorscale: String(runtimeConfig.plotly_config.colorscale || 'Turbo'),
      cmin: colorMin,
      cmax: colorMax,
      colorbar: {
        title: 'UUV z (m)',
        thickness: 14,
        len: 0.75,
      },
    },
    hovertemplate: '关键点<br>x=%{x:.3f}km<br>y=%{y:.3f}km<br>z=%{z:.2f}m<br>UUV深度=%{marker.color:.1f}m<extra></extra>',
  };
}

function toEnemyMarker() {
  const enemy = getCurrentEnemy();

  return {
    type: 'scatter3d',
    mode: 'markers',
    name: '敌方位置标记',
    x: [Number(enemy.enemy_x_km)],
    y: [Number(enemy.enemy_y_km)],
    z: [-Number(enemy.enemy_z_m)],
    marker: {
      size: 8,
      color: '#ff6b6b',
      symbol: 'diamond',
      opacity: 0.95,
    },
    hovertemplate: '敌方位置<br>x=%{x:.3f}km<br>y=%{y:.3f}km<br>z=%{z:.2f}m<extra></extra>',
  };
}

function renderRunMeta(payload) {
  ui.runMeta.innerHTML = '';
  appendMeta(ui.runMeta, '总点对数', String(payload.metadata.total_key_pairs));
  appendMeta(ui.runMeta, '唯一UUV点', String(payload.metadata.uuv_unique_points));
  appendMeta(ui.runMeta, '敌方位置数', String(payload.metadata.enemy_positions_count));
  appendMeta(ui.runMeta, '每敌方点数', String(payload.metadata.points_per_enemy));
  appendMeta(ui.runMeta, '敌方 x(km)', formatValue(payload.stats.enemy_x_km));
  appendMeta(ui.runMeta, '敌方 z(m)', formatValue(payload.stats.enemy_z_m, 1));
  appendMeta(ui.runMeta, 'UUV z最小', formatValue(payload.stats.uuv_z_min_m, 1));
  appendMeta(ui.runMeta, 'UUV z最大', formatValue(payload.stats.uuv_z_max_m, 1));
}

function renderEnemyMeta() {
  const payload = state.payload;
  const enemy = getCurrentEnemy();

  ui.enemyMeta.innerHTML = '';
  appendMeta(ui.enemyMeta, '敌方索引', String(enemy.index));
  appendMeta(ui.enemyMeta, '敌方 x(km)', formatValue(enemy.enemy_x_km));
  appendMeta(ui.enemyMeta, '敌方 y(km)', formatValue(enemy.enemy_y_km));
  appendMeta(ui.enemyMeta, '敌方 z(m)', formatValue(-Number(enemy.enemy_z_m), 1));
  appendMeta(ui.enemyMeta, '当前点云数', String(payload.metadata.uuv_unique_points));
  appendMeta(ui.enemyMeta, '颜色最小', formatValue(payload.stats.color_min_auto, 1));
  appendMeta(ui.enemyMeta, '颜色最大', formatValue(payload.stats.color_max_auto, 1));
  appendMeta(ui.enemyMeta, '地形最大深度', formatValue(payload.stats.terrain_depth_max_m, 1));
}

function renderEnemyList(payload) {
  ui.enemyList.innerHTML = '';
  payload.enemy_positions.forEach((item) => {
    const div = document.createElement('div');
    div.className = 'env-item';
    div.dataset.enemyIndex = String(item.index);
    div.innerHTML = `
      <div class="env-name">敌方 #${item.index + 1}</div>
      <div class="env-detail">x=${Number(item.enemy_x_km).toFixed(3)} km</div>
      <div class="env-detail">y=${Number(item.enemy_y_km).toFixed(3)} km</div>
      <div class="env-detail">z=${(-Number(item.enemy_z_m)).toFixed(1)} m</div>
    `;
    div.addEventListener('click', () => {
      ui.enemySelect.value = String(item.index);
      drawEnemyByIndex(item.index);
    });
    ui.enemyList.appendChild(div);
  });
}

function syncEnemyListActive(enemyIndex) {
  document.querySelectorAll('.env-item').forEach((node) => {
    node.classList.toggle('active', Number(node.dataset.enemyIndex) === enemyIndex);
  });
}

function getSceneLayout() {
  const runtimeConfig = state.runtimeConfig;
  const terrainDepthMax = Number(state.payload.stats.terrain_depth_max_m || 550);

  return {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin: { l: 0, r: 0, t: 6, b: 0 },
    legend: {
      bgcolor: 'rgba(7,17,31,0.72)',
      font: { color: '#e5eefb' },
      x: 0.02,
      y: 0.98,
    },
    scene: {
      bgcolor: 'rgba(0,0,0,0)',
      aspectmode: 'manual',
      aspectratio: {
        x: 1.35,
        y: 1.35,
        z: 0.95,
      },
      camera: {
        eye: {
          x: Number(runtimeConfig.camera_position.x),
          y: Number(runtimeConfig.camera_position.y),
          z: Number(runtimeConfig.camera_position.z),
        },
        center: {
          x: state.cameraCenter.x,
          y: state.cameraCenter.y,
          z: state.cameraCenter.z,
        },
      },
      xaxis: {
        title: 'x (km)',
        color: '#d8e8fb',
        gridcolor: 'rgba(141,173,210,0.25)',
      },
      yaxis: {
        title: 'y (km)',
        color: '#d8e8fb',
        gridcolor: 'rgba(141,173,210,0.25)',
      },
      zaxis: {
        title: 'z (m, 海平面以下为负)',
        color: '#d8e8fb',
        gridcolor: 'rgba(141,173,210,0.25)',
        range: [-terrainDepthMax, 0],
        nticks: 6,
        ticksuffix: ' m',
      },
    },
  };
}

function drawEnemyByIndex(enemyIndex) {
  state.currentEnemyIndex = enemyIndex;
  renderEnemyMeta();
  syncEnemyListActive(enemyIndex);

  const traces = [
    toTerrainTrace(state.payload, state.runtimeConfig),
    toKeyPointTrace(),
    toEnemyMarker(),
  ];

  Plotly.react('plot', traces, getSceneLayout(), {
    responsive: true,
    displaylogo: false,
    scrollZoom: true,
  });

  ui.runBadge.textContent = `当前敌方: #${enemyIndex + 1}`;
  setStatus(`已渲染敌方 #${enemyIndex + 1}，关键点 ${state.payload.metadata.uuv_unique_points}`);
}

function panCameraByKey(key) {
  const panStep = 0.04;
  const nextCenter = {
    ...state.cameraCenter,
  };

  if (key === 'w' || key === 'W') {
    nextCenter.y = clamp(nextCenter.y + panStep, -1, 1);
  } else if (key === 's' || key === 'S') {
    nextCenter.y = clamp(nextCenter.y - panStep, -1, 1);
  } else if (key === 'a' || key === 'A') {
    nextCenter.x = clamp(nextCenter.x - panStep, -1, 1);
  } else if (key === 'd' || key === 'D') {
    nextCenter.x = clamp(nextCenter.x + panStep, -1, 1);
  } else {
    return false;
  }

  state.cameraCenter = nextCenter;
  drawEnemyByIndex(state.currentEnemyIndex);
  setStatus(`相机已平移：x=${nextCenter.x.toFixed(2)}, y=${nextCenter.y.toFixed(2)}`);
  return true;
}

function renderEnemySelect(payload, initialIndex) {
  ui.enemySelect.innerHTML = '';
  payload.enemy_positions.forEach((item) => {
    const option = document.createElement('option');
    option.value = String(item.index);
    option.textContent = item.label;
    ui.enemySelect.appendChild(option);
  });
  ui.enemySelect.value = String(initialIndex);
}

async function loadJson(path) {
  const resp = await fetch(path, { cache: 'no-store' });
  if (!resp.ok) {
    throw new Error(`读取失败: ${path}`);
  }
  return resp.json();
}

async function loadAll() {
  setStatus('正在加载 payload 与 runtime 配置...');
  const [payload, runtimeConfig] = await Promise.all([
    loadJson('./data/payload.json'),
    loadJson('./data/config_runtime.json'),
  ]);

  state.payload = payload;
  state.runtimeConfig = runtimeConfig;
  state.pointSize = Number(runtimeConfig.plotly_config.marker_size || 3.2);
  ui.pointSizeInput.value = String(state.pointSize);

  renderRunMeta(payload);
  renderEnemyList(payload);

  const initialIndex = Number(payload.stats.initial_enemy_index || 0);
  renderEnemySelect(payload, initialIndex);
  drawEnemyByIndex(initialIndex);
}

function bindEvents() {
  ui.reloadBtn.addEventListener('click', () => {
    loadAll().catch((error) => {
      console.error(error);
      ui.runBadge.textContent = '加载失败';
      setStatus(`加载失败: ${error.message}`);
    });
  });

  ui.enemySelect.addEventListener('change', () => {
    const index = Number(ui.enemySelect.value);
    drawEnemyByIndex(index);
  });

  ui.toggleTerrain.addEventListener('change', () => {
    state.showTerrain = ui.toggleTerrain.checked;
    drawEnemyByIndex(state.currentEnemyIndex);
  });

  ui.togglePoints.addEventListener('change', () => {
    state.showPoints = ui.togglePoints.checked;
    drawEnemyByIndex(state.currentEnemyIndex);
  });

  ui.pointSizeInput.addEventListener('change', () => {
    const nextSize = Number(ui.pointSizeInput.value);
    if (Number.isFinite(nextSize) && nextSize > 0) {
      state.pointSize = nextSize;
      drawEnemyByIndex(state.currentEnemyIndex);
    }
  });

  ui.plot.tabIndex = 0;
  ui.plot.setAttribute('aria-label', 'Keypoints 三维视口，点击后使用 WASD 平移视角');
  ui.plot.addEventListener('pointerdown', () => {
    ui.plot.focus();
  });

  window.addEventListener('keydown', (event) => {
    const activeElement = document.activeElement;
    const plotHasFocus = activeElement === ui.plot || ui.plot.contains(activeElement);
    if (!plotHasFocus) {
      return;
    }

    const handled = panCameraByKey(event.key);
    if (handled) {
      event.preventDefault();
    }
  });
}

(async function boot() {
  try {
    ui.runBadge.textContent = '加载中...';
    bindEvents();
    await loadAll();
  } catch (error) {
    console.error(error);
    ui.runBadge.textContent = '加载失败';
    setStatus(`加载失败: ${error.message}`);
  }
})();
