const state = {
  payload: null,
  runtimeConfig: null,
  currentEnemyIndex: 0,
  showTerrain: true,
  showTl: true,
  tlColorCapDb: 120,
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
  toggleTl: document.getElementById('toggleTl'),
  tlCapInput: document.getElementById('tlCapInput'),
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

function formatDb(value) {
  return Number.isFinite(value) ? value.toFixed(3) : '-';
}

function clamp(value, min_value, max_value) {
  return Math.min(max_value, Math.max(min_value, value));
}

function toTerrainTrace(payload, runtimeConfig) {
  const terrain = payload.terrain;
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
    opacity: Number(runtimeConfig.plotly_config.terrain_opacity || 0.42),
    showscale: false,
    hovertemplate: '地形面<br>x=%{x:.3f}km<br>y=%{y:.3f}km<br>z=%{z:.2f}m<extra></extra>',
  };
}

function buildEnemySlice(enemyIndex) {
  const payload = state.payload;
  const runtimeConfig = state.runtimeConfig;
  const tlCapDb = Number.isFinite(Number(state.tlColorCapDb))
    ? Number(state.tlColorCapDb)
    : (Number.isFinite(Number(runtimeConfig.tl_color_cap_db))
      ? Number(runtimeConfig.tl_color_cap_db)
      : 120);

  const points = payload.points;
  const item = payload.tl_by_enemy[enemyIndex];
  const x = [];
  const y = [];
  const z = [];
  const color = [];

  for (let i = 0; i < item.tl_values.length; i += 1) {
    const isValid = Number(item.valid_mask[i]) > 0;
    const tlValue = item.tl_values[i];

    if (!isValid || tlValue === null || !Number.isFinite(Number(tlValue))) {
      continue;
    }

    const tlValueDb = Number(tlValue);
    x.push(Number(points.x_km[i]));
    y.push(Number(points.y_km[i]));
    z.push(-Number(points.z_m[i]));
    color.push(Math.min(tlValueDb, tlCapDb));
  }

  let min = NaN;
  let max = NaN;
  let mean = NaN;
  if (color.length > 0) {
    min = Math.min(...color);
    max = Math.max(...color);
    mean = color.reduce((acc, v) => acc + v, 0) / color.length;
  }

  return {
    item,
    x,
    y,
    z,
    color,
    stats: {
      validCount: color.length,
      min,
      max,
      mean,
    },
  };
}

function toTlTrace(slice) {
  const runtimeConfig = state.runtimeConfig;
  const tlCapDb = Number.isFinite(Number(state.tlColorCapDb))
    ? Number(state.tlColorCapDb)
    : (Number.isFinite(Number(runtimeConfig.tl_color_cap_db))
      ? Number(runtimeConfig.tl_color_cap_db)
      : 120);
  const globalMin = Number.isFinite(Number(runtimeConfig.tl_color_min_db))
    ? Number(runtimeConfig.tl_color_min_db)
    : Number(state.payload.stats.tl_min_db);
  const globalMax = Number.isFinite(Number(runtimeConfig.tl_color_max_db))
    ? Number(runtimeConfig.tl_color_max_db)
    : Number(state.payload.stats.tl_max_db);
  const cappedMax = Number.isFinite(tlCapDb) ? Math.min(globalMax, tlCapDb) : globalMax;

  return {
    type: 'scatter3d',
    mode: 'markers',
    name: '传播损失点云',
    visible: state.showTl,
    x: slice.x,
    y: slice.y,
    z: slice.z,
    marker: {
      size: Number(runtimeConfig.plotly_config.marker_size || 4.0),
      opacity: Number(runtimeConfig.plotly_config.marker_opacity || 0.78),
      color: slice.color,
      colorscale: String(runtimeConfig.plotly_config.colorscale || 'Viridis'),
      cmin: globalMin,
      cmax: cappedMax,
      colorbar: {
        title: 'TL (dB)',
        thickness: 14,
        len: 0.75,
      },
    },
    hovertemplate: 'TL点<br>x=%{x:.3f}km<br>y=%{y:.3f}km<br>z=%{z:.2f}m<br>TL=%{marker.color:.3f}dB<extra></extra>',
  };
}

function toEnemyMarker(slice) {
  const runtimeConfig = state.runtimeConfig;
  const enemyXKm = Number(
    slice.item.enemy_x_km
    ?? runtimeConfig.initial_enemy?.mapped_enemy_x_km
    ?? state.payload.stats?.enemy_x_km
    ?? 2.0,
  );

  return {
    type: 'scatter3d',
    mode: 'markers',
    name: '敌方位置标记',
    x: [enemyXKm],
    y: [Number(slice.item.enemy_y_km)],
    z: [-Number(slice.item.enemy_z_m)],
    marker: {
      size: 7,
      color: '#ff6b6b',
      symbol: 'diamond',
      opacity: 0.95,
    },
    hovertemplate: '敌方位置<br>x=%{x:.3f}km<br>y=%{y:.3f}km<br>z=%{z:.2f}m<extra></extra>',
  };
}

function renderEnemyMeta(slice) {
  ui.enemyMeta.innerHTML = '';
  appendMeta(ui.enemyMeta, '敌方索引', String(slice.item.enemy_index));
  appendMeta(ui.enemyMeta, '敌方 x(km)', Number(slice.item.enemy_x_km ?? 2.0).toFixed(3));
  appendMeta(ui.enemyMeta, '敌方 y(km)', Number(slice.item.enemy_y_km).toFixed(3));
  appendMeta(ui.enemyMeta, '敌方 z(m)', (-Number(slice.item.enemy_z_m)).toFixed(3));
  appendMeta(ui.enemyMeta, '有效点数', String(slice.stats.validCount));
  appendMeta(ui.enemyMeta, 'TL 最小(dB)', formatDb(slice.stats.min));
  appendMeta(ui.enemyMeta, 'TL 最大(dB)', formatDb(slice.stats.max));
  appendMeta(ui.enemyMeta, 'TL 均值(dB)', formatDb(slice.stats.mean));
}

function renderRunMeta(payload) {
  ui.runMeta.innerHTML = '';
  appendMeta(ui.runMeta, '我方点/敌方', String(payload.metadata.our_points_per_enemy));
  appendMeta(ui.runMeta, '敌方位置数', String(payload.metadata.enemy_positions_count));
  appendMeta(ui.runMeta, '敌方 x(km)', formatDb(Number(payload.stats.enemy_x_km)));
  appendMeta(ui.runMeta, 'TL 限幅(dB)', formatDb(Number(state.runtimeConfig?.tl_color_cap_db ?? 120)));
  appendMeta(ui.runMeta, '总点数', String(payload.metadata.total_data_points));
  appendMeta(ui.runMeta, '地形网格', `${payload.terrain.shape[0]} x ${payload.terrain.shape[1]}`);
  appendMeta(ui.runMeta, 'TL全局最小', formatDb(Number(payload.stats.tl_min_db)));
  appendMeta(ui.runMeta, 'TL全局最大', formatDb(Number(payload.stats.tl_max_db)));
}

function renderEnemyList(payload) {
  ui.enemyList.innerHTML = '';
  payload.enemy_positions.forEach((item) => {
    const div = document.createElement('div');
    div.className = 'env-item';
    div.dataset.enemyIndex = String(item.index);
    div.innerHTML = `
      <div class="env-name">敌方 #${item.index + 1}</div>
      <div class="env-detail">x=${Number(item.enemy_x_km ?? 2.0).toFixed(3)} km</div>
      <div class="env-detail">y=${Number(item.enemy_y_km).toFixed(3)} km</div>
      <div class="env-detail">z=${(-Number(item.enemy_z_m)).toFixed(3)} m</div>
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
  const terrainDepthMax = Number(state.payload?.stats?.terrain_depth_max_m || 550);
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

  const slice = buildEnemySlice(enemyIndex);
  renderEnemyMeta(slice);
  syncEnemyListActive(enemyIndex);

  const traces = [
    toTerrainTrace(state.payload, state.runtimeConfig),
    toTlTrace(slice),
    toEnemyMarker(slice),
  ];

  Plotly.react('plot', traces, getSceneLayout(), {
    responsive: true,
    displaylogo: false,
    scrollZoom: true,
  });

  ui.runBadge.textContent = `当前敌方: #${enemyIndex + 1}`;
  setStatus(`已渲染敌方 #${enemyIndex + 1}，有效TL点 ${slice.stats.validCount}`);
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
  state.tlColorCapDb = Number.isFinite(Number(runtimeConfig.tl_color_cap_db))
    ? Number(runtimeConfig.tl_color_cap_db)
    : 120;

  ui.tlCapInput.value = String(state.tlColorCapDb);
  renderRunMeta(payload);
  renderEnemyList(payload);

  const initialIndex = Number(runtimeConfig.initial_enemy.mapped_enemy_index || 0);
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

  ui.toggleTl.addEventListener('change', () => {
    state.showTl = ui.toggleTl.checked;
    drawEnemyByIndex(state.currentEnemyIndex);
  });

  ui.tlCapInput.addEventListener('change', () => {
    const nextCap = Number(ui.tlCapInput.value);
    if (Number.isFinite(nextCap) && nextCap > 0) {
      state.tlColorCapDb = nextCap;
      drawEnemyByIndex(state.currentEnemyIndex);
    }
  });

  ui.plot.tabIndex = 0;
  ui.plot.setAttribute('aria-label', 'TL 三维视口，点击后使用 WASD 平移视角');
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
