const state = {
  payload: null,
  runtimeConfig: null,
  currentFileIndex: 0,
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
  fileSelect: document.getElementById('fileSelect'),
  reloadBtn: document.getElementById('reloadBtn'),
  runBadge: document.getElementById('runBadge'),
  runMeta: document.getElementById('runMeta'),
  fileList: document.getElementById('fileList'),
  fileMeta: document.getElementById('fileMeta'),
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

function buildFileSlice(fileIndex) {
  const item = state.payload.files[fileIndex];
  const runtimeConfig = state.runtimeConfig;
  const tlCapDb = Number.isFinite(Number(state.tlColorCapDb))
    ? Number(state.tlColorCapDb)
    : (Number.isFinite(Number(runtimeConfig.tl_color_cap_db)) ? Number(runtimeConfig.tl_color_cap_db) : 120);

  const x = item.points.x_km.map((v) => Number(v));
  const y = item.points.y_km.map((v) => Number(v));
  const z = item.points.z_m.map((v) => -Number(v));
  const color = item.points.tl_db.map((v) => Math.min(Number(v), tlCapDb));

  return {
    item,
    x,
    y,
    z,
    color,
    stats: {
      validCount: item.stats.point_count,
      min: item.stats.tl_min_db,
      max: item.stats.tl_max_db,
      mean: item.stats.tl_mean_db,
    },
  };
}

function toTlTrace(slice) {
  const runtimeConfig = state.runtimeConfig;
  const tlCapDb = Number.isFinite(Number(state.tlColorCapDb))
    ? Number(state.tlColorCapDb)
    : (Number.isFinite(Number(runtimeConfig.tl_color_cap_db)) ? Number(runtimeConfig.tl_color_cap_db) : 120);
  const globalMin = Number.isFinite(Number(runtimeConfig.tl_color_min_db))
    ? Number(runtimeConfig.tl_color_min_db)
    : Number(state.payload.stats.tl_min_db);
  const globalMax = Number.isFinite(Number(runtimeConfig.tl_color_max_db))
    ? Number(runtimeConfig.tl_color_max_db)
    : Number(state.payload.stats.tl_max_db);
  const cappedMax = Number.isFinite(tlCapDb) ? Math.min(globalMax, tlCapDb) : globalMax;

  const surface = slice.item.surface;
  
  // 将网格数据限幅
  const tlGridCapped = surface.tl_grid.map((row) =>
    row.map((val) => Math.min(Number(val), tlCapDb))
  );
  
  // 深度转为负值（海底在 Z 负方向）
  const zGridNegative = surface.z_grid.map((row) =>
    row.map((val) => -Number(val))
  );

  return {
    type: 'surface',
    name: '传播损失曲面',
    visible: state.showTl,
    x: surface.x_grid.map((v) => Number(v)),
    y: surface.y_grid.map((v) => Number(v)),
    z: zGridNegative,
    surfacecolor: tlGridCapped,
    colorscale: String(runtimeConfig.plotly_config.colorscale || 'Viridis'),
    cmin: globalMin,
    cmax: cappedMax,
    colorbar: {
      title: 'TL (dB)',
      thickness: 14,
      len: 0.75,
    },
    showscale: true,
    opacity: Number(runtimeConfig.plotly_config.marker_opacity || 0.78),
    hovertemplate: 'TL曲面<br>x=%{x:.3f}km<br>y=%{y:.3f}km<br>z=%{z:.2f}m<br>TL=%{surfacecolor:.3f}dB<extra></extra>',
  };
}

function toSourceMarker(slice) {
  return {
    type: 'scatter3d',
    mode: 'markers+text',
    name: '源点位置',
    x: [Number(slice.item.source_position_km[0])],
    y: [Number(slice.item.source_position_km[1])],
    z: [-Number(slice.item.source_position_km[2])],
    text: ['Source'],
    textposition: 'top center',
    marker: {
      size: 7,
      color: '#4cc9f0',
      symbol: 'diamond',
      opacity: 0.95,
    },
    hovertemplate: '源点<br>x=%{x:.3f}km<br>y=%{y:.3f}km<br>z=%{z:.2f}m<extra></extra>',
  };
}

function toEnemyMarker(slice) {
  return {
    type: 'scatter3d',
    mode: 'markers+text',
    name: '敌方位置',
    x: [Number(slice.item.enemy_position_km[0])],
    y: [Number(slice.item.enemy_position_km[1])],
    z: [-Number(slice.item.enemy_position_km[2])],
    text: ['Enemy'],
    textposition: 'top center',
    marker: {
      size: 7,
      color: '#ff6b6b',
      symbol: 'diamond',
      opacity: 0.95,
    },
    hovertemplate: '敌方<br>x=%{x:.3f}km<br>y=%{y:.3f}km<br>z=%{z:.2f}m<extra></extra>',
  };
}

function renderFileMeta(slice) {
  ui.fileMeta.innerHTML = '';
  appendMeta(ui.fileMeta, '文件名', slice.item.file_name);
  appendMeta(ui.fileMeta, '点数量', String(slice.stats.validCount));
  appendMeta(ui.fileMeta, 'TL 最小(dB)', formatDb(Number(slice.stats.min)));
  appendMeta(ui.fileMeta, 'TL 最大(dB)', formatDb(Number(slice.stats.max)));
  appendMeta(ui.fileMeta, 'TL 均值(dB)', formatDb(Number(slice.stats.mean)));
  appendMeta(ui.fileMeta, '源点(m)', `${slice.item.source_position_m.map((v) => Number(v).toFixed(2)).join(', ')}`);
  appendMeta(ui.fileMeta, '敌方(m)', `${slice.item.enemy_position_m.map((v) => Number(v).toFixed(2)).join(', ')}`);
}

function renderRunMeta(payload) {
  ui.runMeta.innerHTML = '';
  appendMeta(ui.runMeta, '文件数量', String(payload.metadata.file_count));
  appendMeta(ui.runMeta, '总点数', String(payload.metadata.total_point_count));
  appendMeta(ui.runMeta, 'TL全局最小', formatDb(Number(payload.stats.tl_min_db)));
  appendMeta(ui.runMeta, 'TL全局最大', formatDb(Number(payload.stats.tl_max_db)));
  appendMeta(ui.runMeta, 'TL 限幅(dB)', formatDb(Number(state.runtimeConfig?.tl_color_cap_db ?? 120)));
  appendMeta(ui.runMeta, '地形网格', `${payload.terrain.shape[0]} x ${payload.terrain.shape[1]}`);
}

function renderFileList(payload) {
  ui.fileList.innerHTML = '';
  payload.files.forEach((item) => {
    const div = document.createElement('div');
    div.className = 'env-item';
    div.dataset.fileIndex = String(item.file_index);
    div.innerHTML = `
      <div class="env-name">${item.label}</div>
      <div class="env-detail">点数: ${item.stats.point_count}</div>
      <div class="env-detail">TL: ${Number(item.stats.tl_min_db).toFixed(2)} ~ ${Number(item.stats.tl_max_db).toFixed(2)} dB</div>
    `;
    div.addEventListener('click', () => {
      ui.fileSelect.value = String(item.file_index);
      drawFileByIndex(item.file_index);
    });
    ui.fileList.appendChild(div);
  });
}

function syncFileListActive(fileIndex) {
  document.querySelectorAll('.env-item').forEach((node) => {
    node.classList.toggle('active', Number(node.dataset.fileIndex) === fileIndex);
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

function drawFileByIndex(fileIndex) {
  state.currentFileIndex = fileIndex;

  const slice = buildFileSlice(fileIndex);
  renderFileMeta(slice);
  syncFileListActive(fileIndex);

  const traces = [
    toTerrainTrace(state.payload, state.runtimeConfig),
    toTlTrace(slice),
    toSourceMarker(slice),
    toEnemyMarker(slice),
  ];

  Plotly.react('plot', traces, getSceneLayout(), {
    responsive: true,
    displaylogo: false,
    scrollZoom: true,
  });

  ui.runBadge.textContent = `当前文件: #${fileIndex + 1}`;
  setStatus(`已渲染文件 #${fileIndex + 1}，有效TL点 ${slice.stats.validCount}`);
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
  drawFileByIndex(state.currentFileIndex);
  setStatus(`相机已平移：x=${nextCenter.x.toFixed(2)}, y=${nextCenter.y.toFixed(2)}`);
  return true;
}

function renderFileSelect(payload, initialIndex) {
  ui.fileSelect.innerHTML = '';
  payload.files.forEach((item) => {
    const option = document.createElement('option');
    option.value = String(item.file_index);
    option.textContent = item.label;
    ui.fileSelect.appendChild(option);
  });
  ui.fileSelect.value = String(initialIndex);
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
  renderFileList(payload);

  const initialIndex = Number(runtimeConfig.initial_file_index || 0);
  renderFileSelect(payload, initialIndex);
  drawFileByIndex(initialIndex);
}

function bindEvents() {
  ui.reloadBtn.addEventListener('click', () => {
    loadAll().catch((error) => {
      console.error(error);
      ui.runBadge.textContent = '加载失败';
      setStatus(`加载失败: ${error.message}`);
    });
  });

  ui.fileSelect.addEventListener('change', () => {
    const index = Number(ui.fileSelect.value);
    drawFileByIndex(index);
  });

  ui.toggleTerrain.addEventListener('change', () => {
    state.showTerrain = ui.toggleTerrain.checked;
    drawFileByIndex(state.currentFileIndex);
  });

  ui.toggleTl.addEventListener('change', () => {
    state.showTl = ui.toggleTl.checked;
    drawFileByIndex(state.currentFileIndex);
  });

  ui.tlCapInput.addEventListener('change', () => {
    const nextCap = Number(ui.tlCapInput.value);
    if (Number.isFinite(nextCap) && nextCap > 0) {
      state.tlColorCapDb = nextCap;
      drawFileByIndex(state.currentFileIndex);
    }
  });

  ui.plot.tabIndex = 0;
  ui.plot.setAttribute('aria-label', 'C++ TL 三维视口，点击后使用 WASD 平移视角');
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
