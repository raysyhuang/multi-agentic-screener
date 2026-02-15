/* Dashboard SPA — vanilla JS with tab routing and lazy data loading */

(function () {
  'use strict';

  // -----------------------------------------------------------------------
  // State
  // -----------------------------------------------------------------------
  const loaded = { signals: false, performance: false, charts: false, compare: false, pipeline: false, costs: false };
  let equityChart = null;
  let drawdownChart = null;
  let calibrationChart = null;

  // -----------------------------------------------------------------------
  // Tab Router
  // -----------------------------------------------------------------------
  document.querySelectorAll('.nav-tab').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var tab = btn.dataset.tab;
      switchTab(tab);
    });
  });

  function switchTab(tab) {
    document.querySelectorAll('.nav-tab').forEach(function (b) { b.classList.remove('active'); });
    document.querySelectorAll('.view').forEach(function (v) { v.classList.remove('active'); });

    var btn = document.querySelector('[data-tab="' + tab + '"]');
    if (btn) btn.classList.add('active');
    var view = document.getElementById(tab + '-view');
    if (view) view.classList.add('active');

    if (!loaded[tab]) {
      loaded[tab] = true;
      loadTab(tab);
    }
  }

  function loadTab(tab) {
    switch (tab) {
      case 'signals': loadSignals(); break;
      case 'performance': loadPerformance(); break;
      case 'charts': loadCharts(); break;
      case 'compare': loadCompare(); break;
      case 'pipeline': loadPipeline(); break;
      case 'costs': loadCosts(); break;
    }
  }

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------
  function fetchJSON(url) {
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    });
  }

  function showSpinner(viewId) {
    document.getElementById(viewId).innerHTML = '<div class="spinner">Loading...</div>';
  }

  function showEmpty(viewId, message) {
    document.getElementById(viewId).innerHTML =
      '<div class="empty-state"><h3>No Data</h3><p>' + message + '</p></div>';
  }

  function fmt(n, decimals) {
    if (n == null) return '—';
    return Number(n).toFixed(decimals != null ? decimals : 2);
  }

  function fmtPct(n) {
    if (n == null) return '—';
    var s = Number(n).toFixed(2);
    return (n > 0 ? '+' : '') + s + '%';
  }

  function confidenceColor(c) {
    if (c >= 70) return 'var(--green)';
    if (c >= 50) return 'var(--yellow)';
    return 'var(--red)';
  }

  function regimeBadge(regime) {
    var r = (regime || '').toLowerCase();
    var cls = 'badge badge-' + r;
    return '<span class="' + cls + '">' + regime + '</span>';
  }

  function createChart(containerId, height) {
    var el = document.getElementById(containerId);
    if (!el || typeof LightweightCharts === 'undefined') return null;

    return LightweightCharts.createChart(el, {
      width: el.clientWidth,
      height: height || 300,
      layout: {
        background: { color: '#1a1d29' },
        textColor: '#71717a',
      },
      grid: {
        vertLines: { color: '#2a2d3a' },
        horzLines: { color: '#2a2d3a' },
      },
      rightPriceScale: { borderColor: '#2a2d3a' },
      timeScale: { borderColor: '#2a2d3a' },
    });
  }

  // -----------------------------------------------------------------------
  // Signals Tab
  // -----------------------------------------------------------------------
  function loadSignals() {
    showSpinner('signals-view');
    fetchJSON('/api/dashboard/signals').then(function (data) {
      var view = document.getElementById('signals-view');
      if (!data.signals || data.signals.length === 0) {
        showEmpty('signals-view', 'No signals from the latest pipeline run.');
        return;
      }

      var header = '<div class="card" style="margin-bottom:1.5rem">' +
        '<div class="card-header">' +
        '<span class="card-title">Latest Signals — ' + (data.run_date || '?') + '</span>' +
        regimeBadge(data.regime || '') +
        '</div></div>';

      var cards = data.signals.map(renderSignalCard).join('');
      view.innerHTML = header + '<div class="signals-grid">' + cards + '</div>';
    }).catch(function () {
      showEmpty('signals-view', 'Failed to load signals.');
    });
  }

  function renderSignalCard(s) {
    var dirClass = (s.direction || '').toLowerCase() === 'long' ? 'direction-long' : 'direction-short';
    var conf = s.confidence || 0;

    return '<div class="card signal-card">' +
      '<div class="card-header">' +
        '<div>' +
          '<span class="signal-ticker">' + s.ticker + '</span>' +
          '<span class="direction-badge ' + dirClass + '">' + s.direction + '</span>' +
          '<span class="signal-model">' + (s.signal_model || '') + '</span>' +
        '</div>' +
      '</div>' +
      '<div class="confidence-label">Confidence: ' + fmt(conf, 0) + '/100</div>' +
      '<div class="confidence-meter"><div class="confidence-fill" style="width:' + conf + '%;background:' + confidenceColor(conf) + '"></div></div>' +
      '<div class="trade-grid">' +
        gridItem('Entry', '$' + fmt(s.entry_price)) +
        gridItem('Stop', '$' + fmt(s.stop_loss), 'var(--red)') +
        gridItem('Target', '$' + fmt(s.target_1), 'var(--green)') +
        gridItem('Hold', (s.holding_period_days || '—') + 'd') +
      '</div>' +
      (s.thesis ? '<div class="signal-thesis">' + s.thesis + '</div>' : '') +
    '</div>';
  }

  function gridItem(label, value, color) {
    var style = color ? ' style="color:' + color + '"' : '';
    return '<div class="trade-grid-item">' +
      '<div class="label">' + label + '</div>' +
      '<div class="value"' + style + '>' + value + '</div>' +
    '</div>';
  }

  // -----------------------------------------------------------------------
  // Performance Tab
  // -----------------------------------------------------------------------
  function loadPerformance() {
    showSpinner('performance-view');
    fetchJSON('/api/dashboard/performance').then(function (data) {
      var view = document.getElementById('performance-view');

      if (!data.total_signals || data.total_signals === 0) {
        showEmpty('performance-view', 'No closed trades yet.');
        return;
      }

      var overall = data.overall || {};
      var risk = data.risk_metrics || {};

      var html = '<div class="metrics-grid">' +
        metricCard(data.total_signals, 'Total Trades') +
        metricCard(fmtPct(overall.win_rate * 100), 'Win Rate', overall.win_rate >= 0.5) +
        metricCard(fmtPct(overall.avg_pnl), 'Avg P&L', overall.avg_pnl > 0) +
        metricCard(fmt(risk.sharpe_ratio), 'Sharpe Ratio', risk.sharpe_ratio > 0) +
        metricCard(fmt(risk.sortino_ratio), 'Sortino Ratio', risk.sortino_ratio > 0) +
        metricCard(fmtPct(risk.max_drawdown_pct), 'Max Drawdown', false) +
        metricCard(fmt(risk.profit_factor), 'Profit Factor', risk.profit_factor > 1) +
        metricCard(fmt(risk.expectancy), 'Expectancy', risk.expectancy > 0) +
      '</div>';

      // Equity curve
      if (data.equity_curve && data.equity_curve.length > 0) {
        html += '<div class="chart-container">' +
          '<div class="chart-title">Cumulative P&L (Equity Curve)</div>' +
          '<div id="equity-chart"></div>' +
        '</div>';
      }

      // Breakdown tables
      if (data.by_model && Object.keys(data.by_model).length > 0) {
        html += breakdownTable('By Signal Model', data.by_model);
      }
      if (data.by_regime && Object.keys(data.by_regime).length > 0) {
        html += breakdownTable('By Regime', data.by_regime);
      }

      view.innerHTML = html;

      // Render chart after DOM insert
      if (data.equity_curve && data.equity_curve.length > 0) {
        renderEquityCurve(data.equity_curve);
      }
    }).catch(function () {
      showEmpty('performance-view', 'Failed to load performance data.');
    });
  }

  function metricCard(value, label, isPositive) {
    var cls = '';
    if (isPositive === true) cls = ' positive';
    else if (isPositive === false && value !== '—') cls = ' negative';
    return '<div class="metric-card">' +
      '<div class="metric-value' + cls + '">' + value + '</div>' +
      '<div class="metric-label">' + label + '</div>' +
    '</div>';
  }

  function breakdownTable(title, data) {
    var rows = Object.keys(data).map(function (key) {
      var d = data[key];
      var pnlClass = d.avg_pnl > 0 ? 'positive' : (d.avg_pnl < 0 ? 'negative' : '');
      return '<tr>' +
        '<td>' + key + '</td>' +
        '<td>' + d.trades + '</td>' +
        '<td>' + fmtPct(d.win_rate * 100) + '</td>' +
        '<td class="' + pnlClass + '">' + fmtPct(d.avg_pnl) + '</td>' +
      '</tr>';
    }).join('');

    return '<div class="card">' +
      '<div class="card-title" style="margin-bottom:0.75rem">' + title + '</div>' +
      '<table class="data-table">' +
        '<thead><tr><th>Name</th><th>Trades</th><th>Win Rate</th><th>Avg P&L</th></tr></thead>' +
        '<tbody>' + rows + '</tbody>' +
      '</table>' +
    '</div>';
  }

  function renderEquityCurve(data) {
    var el = document.getElementById('equity-chart');
    if (!el) return;

    if (typeof LightweightCharts === 'undefined') {
      el.innerHTML = '<div class="empty-state"><p>Chart library unavailable.</p></div>';
      return;
    }

    if (equityChart) {
      equityChart.remove();
      equityChart = null;
    }

    var chart = LightweightCharts.createChart(el, {
      width: el.clientWidth,
      height: 350,
      layout: {
        background: { color: '#1a1d29' },
        textColor: '#71717a',
      },
      grid: {
        vertLines: { color: '#2a2d3a' },
        horzLines: { color: '#2a2d3a' },
      },
      rightPriceScale: { borderColor: '#2a2d3a' },
      timeScale: { borderColor: '#2a2d3a' },
    });
    equityChart = chart;

    var series = chart.addLineSeries({
      color: '#3b82f6',
      lineWidth: 2,
    });

    var chartData = data.map(function (p) {
      return { time: p.time, value: p.value };
    });
    series.setData(chartData);
    chart.timeScale().fitContent();

    window.addEventListener('resize', function () {
      if (equityChart && el.clientWidth > 0) {
        equityChart.applyOptions({ width: el.clientWidth });
      }
    });
  }

  // -----------------------------------------------------------------------
  // Charts Tab (PR4: 5 statistical charts)
  // -----------------------------------------------------------------------
  function loadCharts() {
    showSpinner('charts-view');

    Promise.all([
      fetchJSON('/api/dashboard/equity-curve?days=90'),
      fetchJSON('/api/dashboard/drawdown?days=90'),
      fetchJSON('/api/dashboard/return-distribution?days=90'),
      fetchJSON('/api/dashboard/regime-matrix?days=180'),
      fetchJSON('/api/dashboard/calibration'),
    ]).then(function (results) {
      var equity = results[0];
      var drawdown = results[1];
      var distribution = results[2];
      var regimeMatrix = results[3];
      var calibration = results[4];
      var view = document.getElementById('charts-view');

      var html = '';

      // 1. Equity Curve
      if (equity.equity_curve && equity.equity_curve.length > 0) {
        html += '<div class="chart-container">' +
          '<div class="chart-title">Equity Curve (Cumulative Returns)</div>' +
          '<div id="charts-equity"></div>' +
        '</div>';
      } else {
        html += '<div class="card"><div class="empty-state"><p>No equity curve data yet.</p></div></div>';
      }

      // 2. Drawdown Curve
      if (drawdown.drawdown && drawdown.drawdown.length > 0) {
        html += '<div class="chart-container">' +
          '<div class="chart-title">Drawdown</div>' +
          '<div id="charts-drawdown"></div>' +
        '</div>';
      } else {
        html += '<div class="card"><div class="empty-state"><p>No drawdown data yet.</p></div></div>';
      }

      // 3. Return Distribution by Model
      if (distribution.distribution && Object.keys(distribution.distribution).length > 0) {
        html += '<div class="card">' +
          '<div class="card-title" style="margin-bottom:0.75rem">Return Distribution by Model</div>';
        Object.keys(distribution.distribution).forEach(function (model) {
          var d = distribution.distribution[model];
          var histogram = renderHistogram(d.returns);
          html += '<div style="margin-bottom:1rem">' +
            '<div style="font-weight:600;margin-bottom:0.25rem">' + model +
            ' <span style="color:#71717a">(n=' + d.count + ', mean=' + fmtPct(d.mean) + ', std=' + fmt(d.std) + ')</span></div>' +
            histogram +
          '</div>';
        });
        html += '</div>';
      } else {
        html += '<div class="card"><div class="empty-state"><p>No return distribution data yet.</p></div></div>';
      }

      // 4. Regime Matrix
      if (regimeMatrix.matrix && regimeMatrix.matrix.length > 0) {
        html += renderRegimeMatrix(regimeMatrix.matrix);
      } else {
        html += '<div class="card"><div class="empty-state"><p>No regime matrix data yet.</p></div></div>';
      }

      // 5. Calibration Curve
      if (calibration.calibration && calibration.calibration.length > 0) {
        html += renderCalibrationTable(calibration.calibration);
      } else {
        html += '<div class="card"><div class="empty-state"><p>No calibration data yet.</p></div></div>';
      }

      view.innerHTML = html;

      // Render LightweightCharts after DOM
      if (equity.equity_curve && equity.equity_curve.length > 0) {
        renderTimeChart('charts-equity', equity.equity_curve, '#3b82f6', 'line');
      }
      if (drawdown.drawdown && drawdown.drawdown.length > 0) {
        renderTimeChart('charts-drawdown', drawdown.drawdown, '#ef4444', 'area');
      }
    }).catch(function () {
      showEmpty('charts-view', 'Failed to load chart data.');
    });
  }

  function renderTimeChart(containerId, data, color, type) {
    var el = document.getElementById(containerId);
    if (!el || typeof LightweightCharts === 'undefined') return;

    var chart = LightweightCharts.createChart(el, {
      width: el.clientWidth,
      height: 300,
      layout: { background: { color: '#1a1d29' }, textColor: '#71717a' },
      grid: { vertLines: { color: '#2a2d3a' }, horzLines: { color: '#2a2d3a' } },
      rightPriceScale: { borderColor: '#2a2d3a' },
      timeScale: { borderColor: '#2a2d3a' },
    });

    var series;
    if (type === 'area') {
      series = chart.addAreaSeries({
        lineColor: color,
        topColor: color + '80',
        bottomColor: color + '10',
        lineWidth: 2,
      });
    } else {
      series = chart.addLineSeries({ color: color, lineWidth: 2 });
    }

    series.setData(data.map(function (p) { return { time: p.time, value: p.value }; }));
    chart.timeScale().fitContent();

    window.addEventListener('resize', function () {
      if (el.clientWidth > 0) chart.applyOptions({ width: el.clientWidth });
    });
  }

  function renderHistogram(returns) {
    if (!returns || returns.length === 0) return '<p>No data</p>';
    // Simple text-based histogram using buckets
    var buckets = {};
    var step = 1; // 1% buckets
    returns.forEach(function (r) {
      var bucket = Math.floor(r / step) * step;
      var key = bucket + '%';
      buckets[key] = (buckets[key] || 0) + 1;
    });

    var maxCount = Math.max.apply(null, Object.values(buckets));
    var rows = Object.keys(buckets).sort(function (a, b) {
      return parseFloat(a) - parseFloat(b);
    }).map(function (key) {
      var count = buckets[key];
      var width = Math.round(count / maxCount * 100);
      var color = parseFloat(key) >= 0 ? 'var(--green)' : 'var(--red)';
      return '<div style="display:flex;align-items:center;gap:0.5rem;margin:2px 0">' +
        '<span style="min-width:50px;text-align:right;font-size:0.8rem">' + key + '</span>' +
        '<div style="background:' + color + ';height:16px;width:' + width + '%;border-radius:2px;min-width:2px"></div>' +
        '<span style="font-size:0.75rem;color:#71717a">' + count + '</span>' +
      '</div>';
    }).join('');

    return '<div style="max-width:400px">' + rows + '</div>';
  }

  function renderRegimeMatrix(matrix) {
    // Group by model
    var models = {};
    var regimes = new Set();
    matrix.forEach(function (m) {
      if (!models[m.model]) models[m.model] = {};
      models[m.model][m.regime] = m;
      regimes.add(m.regime);
    });

    var regimeList = Array.from(regimes).sort();
    var headerCells = regimeList.map(function (r) { return '<th>' + r + '</th>'; }).join('');

    var rows = Object.keys(models).map(function (model) {
      var cells = regimeList.map(function (regime) {
        var cell = models[model][regime];
        if (!cell) return '<td style="color:#71717a">—</td>';
        var wr = cell.win_rate * 100;
        var bg = wr >= 55 ? 'rgba(34,197,94,0.2)' : (wr < 45 ? 'rgba(239,68,68,0.2)' : 'rgba(234,179,8,0.2)');
        return '<td style="background:' + bg + '">' +
          '<div>' + fmtPct(wr) + '</div>' +
          '<div style="font-size:0.7rem;color:#71717a">' + cell.trades + ' trades</div>' +
        '</td>';
      }).join('');
      return '<tr><td><b>' + model + '</b></td>' + cells + '</tr>';
    }).join('');

    return '<div class="card">' +
      '<div class="card-title" style="margin-bottom:0.75rem">Win Rate by Model x Regime</div>' +
      '<table class="data-table">' +
        '<thead><tr><th>Model</th>' + headerCells + '</tr></thead>' +
        '<tbody>' + rows + '</tbody>' +
      '</table>' +
    '</div>';
  }

  function renderCalibrationTable(data) {
    var rows = data.map(function (b) {
      var error = b.calibration_error;
      var errClass = error > 0.15 ? 'negative' : (error < 0.05 ? 'positive' : '');
      return '<tr>' +
        '<td>' + b.bucket + '</td>' +
        '<td>' + b.trades + '</td>' +
        '<td>' + fmtPct(b.expected_win_rate * 100) + '</td>' +
        '<td>' + fmtPct(b.actual_win_rate * 100) + '</td>' +
        '<td class="' + errClass + '">' + fmt(error, 4) + '</td>' +
        '<td>' + fmtPct(b.avg_pnl) + '</td>' +
      '</tr>';
    }).join('');

    return '<div class="card">' +
      '<div class="card-title" style="margin-bottom:0.75rem">Confidence Calibration</div>' +
      '<table class="data-table">' +
        '<thead><tr><th>Bucket</th><th>Trades</th><th>Expected WR</th><th>Actual WR</th><th>Cal. Error</th><th>Avg P&L</th></tr></thead>' +
        '<tbody>' + rows + '</tbody>' +
      '</table>' +
    '</div>';
  }

  // -----------------------------------------------------------------------
  // Compare Tab (PR4: LLM uplift — mode comparison)
  // -----------------------------------------------------------------------
  function loadCompare() {
    showSpinner('compare-view');
    fetchJSON('/api/dashboard/mode-comparison').then(function (data) {
      var view = document.getElementById('compare-view');

      if (!data.comparison || data.comparison.length === 0) {
        showEmpty('compare-view', 'No mode comparison data. Run the pipeline in different modes first.');
        return;
      }

      var html = '<div class="card">' +
        '<div class="card-title" style="margin-bottom:0.75rem">LLM Uplift: Mode Comparison</div>' +
        '<p style="color:#71717a;margin-bottom:1rem">Compare quant_only vs hybrid vs agentic_full performance</p>';

      // Metrics grid per mode
      html += '<div class="metrics-grid">';
      data.comparison.forEach(function (m) {
        var pnlClass = m.avg_pnl > 0 ? 'positive' : (m.avg_pnl < 0 ? 'negative' : '');
        html += '<div class="metric-card" style="border-left:3px solid ' +
          (m.mode === 'agentic_full' ? '#3b82f6' : (m.mode === 'hybrid' ? '#a855f7' : '#22c55e')) + '">' +
          '<div class="metric-value">' + (m.mode || 'unknown').toUpperCase() + '</div>' +
          '<div class="metric-label">' + m.trades + ' trades</div>' +
          '<div class="metric-label ' + pnlClass + '">WR: ' + fmtPct(m.win_rate * 100) +
          ' | Avg: ' + fmtPct(m.avg_pnl) +
          ' | Total: ' + fmtPct(m.total_return) + '</div>' +
        '</div>';
      });
      html += '</div>';

      // Comparison table
      var rows = data.comparison.map(function (m) {
        var pnlClass = m.avg_pnl > 0 ? 'positive' : (m.avg_pnl < 0 ? 'negative' : '');
        return '<tr>' +
          '<td><b>' + (m.mode || '?').toUpperCase() + '</b></td>' +
          '<td>' + m.trades + '</td>' +
          '<td>' + fmtPct(m.win_rate * 100) + '</td>' +
          '<td class="' + pnlClass + '">' + fmtPct(m.avg_pnl) + '</td>' +
          '<td class="' + pnlClass + '">' + fmtPct(m.total_return) + '</td>' +
        '</tr>';
      }).join('');

      html += '<table class="data-table">' +
        '<thead><tr><th>Mode</th><th>Trades</th><th>Win Rate</th><th>Avg P&L</th><th>Total Return</th></tr></thead>' +
        '<tbody>' + rows + '</tbody>' +
      '</table></div>';

      view.innerHTML = html;
    }).catch(function () {
      showEmpty('compare-view', 'Failed to load mode comparison data.');
    });
  }

  // -----------------------------------------------------------------------
  // Pipeline Tab
  // -----------------------------------------------------------------------
  function loadPipeline() {
    showSpinner('pipeline-view');
    fetchJSON('/api/runs?limit=30').then(function (runs) {
      var view = document.getElementById('pipeline-view');

      if (!runs || runs.length === 0) {
        showEmpty('pipeline-view', 'No pipeline runs recorded yet.');
        return;
      }

      var rows = runs.map(function (r) {
        return '<tr class="run-row" data-run-date="' + r.run_date + '">' +
          '<td>' + r.run_date + '</td>' +
          '<td>' + regimeBadge(r.regime) + '</td>' +
          '<td>' + (r.universe_size || '—') + '</td>' +
          '<td>' + (r.candidates_scored || '—') + '</td>' +
          '<td>' + (r.pipeline_duration_s != null ? fmt(r.pipeline_duration_s, 1) + 's' : '—') + '</td>' +
        '</tr>';
      }).join('');

      view.innerHTML = '<div class="card">' +
        '<div class="card-title" style="margin-bottom:0.75rem">Pipeline Runs</div>' +
        '<table class="data-table">' +
          '<thead><tr><th>Date</th><th>Regime</th><th>Universe</th><th>Scored</th><th>Duration</th></tr></thead>' +
          '<tbody>' + rows + '</tbody>' +
        '</table>' +
      '</div>' +
      '<div id="run-detail"></div>';
    }).catch(function () {
      showEmpty('pipeline-view', 'Failed to load pipeline runs.');
    });
  }

  // -----------------------------------------------------------------------
  // Costs Tab
  // -----------------------------------------------------------------------
  function loadCosts() {
    showSpinner('costs-view');

    Promise.all([
      fetchJSON('/api/costs?days=30'),
      fetchJSON('/api/cache-stats'),
    ]).then(function (results) {
      var costs = results[0];
      var cache = results[1];
      var view = document.getElementById('costs-view');

      var html = '<div class="metrics-grid">' +
        metricCard('$' + fmt(costs.total_cost_usd, 4), 'Total Cost (30d)') +
        metricCard(numberFormat(costs.total_tokens_in), 'Tokens In') +
        metricCard(numberFormat(costs.total_tokens_out), 'Tokens Out') +
        metricCard(fmtPct(cache.hit_rate * 100), 'Cache Hit Rate', cache.hit_rate > 0.5) +
        metricCard(cache.hits, 'Cache Hits') +
        metricCard(cache.misses, 'Cache Misses') +
        metricCard(cache.total_entries, 'Cached Entries') +
        metricCard(cache.evictions, 'Evictions') +
      '</div>';

      // By-agent breakdown
      if (costs.by_agent && Object.keys(costs.by_agent).length > 0) {
        var agentRows = Object.keys(costs.by_agent).map(function (agent) {
          return '<tr><td>' + agent + '</td><td>$' + fmt(costs.by_agent[agent], 4) + '</td></tr>';
        }).join('');

        html += '<div class="card">' +
          '<div class="card-title" style="margin-bottom:0.75rem">Cost by Agent</div>' +
          '<table class="data-table">' +
            '<thead><tr><th>Agent</th><th>Cost (USD)</th></tr></thead>' +
            '<tbody>' + agentRows + '</tbody>' +
          '</table>' +
        '</div>';
      }

      // Daily breakdown
      if (costs.by_date && Object.keys(costs.by_date).length > 0) {
        var dateRows = Object.keys(costs.by_date).sort().reverse().map(function (d) {
          return '<tr><td>' + d + '</td><td>$' + fmt(costs.by_date[d], 4) + '</td></tr>';
        }).join('');

        html += '<div class="card">' +
          '<div class="card-title" style="margin-bottom:0.75rem">Cost by Date</div>' +
          '<table class="data-table">' +
            '<thead><tr><th>Date</th><th>Cost (USD)</th></tr></thead>' +
            '<tbody>' + dateRows + '</tbody>' +
          '</table>' +
        '</div>';
      }

      view.innerHTML = html;
    }).catch(function () {
      showEmpty('costs-view', 'Failed to load cost data.');
    });
  }

  function numberFormat(n) {
    if (n == null) return '—';
    return Number(n).toLocaleString();
  }

  // -----------------------------------------------------------------------
  // Boot: load default tab
  // -----------------------------------------------------------------------
  switchTab('signals');

})();
