/* Dashboard SPA — vanilla JS with tab routing and lazy data loading */

(function () {
  'use strict';

  // -----------------------------------------------------------------------
  // State
  // -----------------------------------------------------------------------
  const loaded = { signals: false, performance: false, pipeline: false, costs: false };
  let equityChart = null;

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
