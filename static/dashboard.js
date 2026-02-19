/* Dashboard SPA — vanilla JS with tab routing and lazy data loading */

(function () {
  'use strict';

  // -----------------------------------------------------------------------
  // Theme-aware chart colors
  // -----------------------------------------------------------------------
  function chartColors() {
    var dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    return {
      bg: dark ? '#1e293b' : '#ffffff',
      text: dark ? '#94a3b8' : '#64748b',
      grid: dark ? '#334155' : '#e2e8f0',
      border: dark ? '#334155' : '#e2e8f0',
      teal: '#14b8a6',
      red: '#ef4444',
      green: '#22c55e',
    };
  }

  // -----------------------------------------------------------------------
  // State
  // -----------------------------------------------------------------------
  var loaded = {
    signals: false, crossengine: false, performance: false,
    charts: false, compare: false, pipeline: false, costs: false
  };
  var equityChart = null;

  // -----------------------------------------------------------------------
  // Mobile Hamburger Menu
  // -----------------------------------------------------------------------
  var hamburger = document.querySelector('.nav-hamburger');
  var navTabs = document.querySelector('.nav-tabs');
  var activeLabel = document.querySelector('.nav-active-label');

  if (hamburger) {
    hamburger.addEventListener('click', function () {
      var expanded = hamburger.getAttribute('aria-expanded') === 'true';
      hamburger.setAttribute('aria-expanded', String(!expanded));
      navTabs.classList.toggle('open');
    });
  }

  // -----------------------------------------------------------------------
  // Tab Router
  // -----------------------------------------------------------------------
  document.querySelectorAll('.nav-tab').forEach(function (btn) {
    btn.addEventListener('click', function () {
      switchTab(btn.dataset.tab);
    });
  });

  function switchTab(tab) {
    document.querySelectorAll('.nav-tab').forEach(function (b) { b.classList.remove('active'); });
    document.querySelectorAll('.view').forEach(function (v) { v.classList.remove('active'); });

    var btn = document.querySelector('[data-tab="' + tab + '"]');
    if (btn) {
      btn.classList.add('active');
      if (activeLabel) activeLabel.textContent = btn.textContent;
    }
    var view = document.getElementById(tab + '-view');
    if (view) view.classList.add('active');

    // Close mobile menu after selection
    if (hamburger && navTabs) {
      hamburger.setAttribute('aria-expanded', 'false');
      navTabs.classList.remove('open');
    }

    if (!loaded[tab]) {
      loaded[tab] = true;
      loadTab(tab);
    }
  }

  function loadTab(tab) {
    switch (tab) {
      case 'signals': loadSignals(); break;
      case 'crossengine': loadCrossEngine(); break;
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
    if (n == null) return '\u2014';
    return Number(n).toFixed(decimals != null ? decimals : 2);
  }

  function fmtPct(n) {
    if (n == null) return '\u2014';
    var s = Number(n).toFixed(2);
    return (n > 0 ? '+' : '') + s + '%';
  }

  function confidenceColor(c) {
    if (c >= 70) return 'var(--green)';
    if (c >= 50) return 'var(--amber)';
    return 'var(--red)';
  }

  function regimeBadge(regime) {
    var r = (regime || '').toLowerCase();
    var cls = 'badge badge-' + r;
    return '<span class="' + cls + '">' + regime + '</span>';
  }

  function factorClass(value) {
    if (value >= 0.8) return 'good';
    if (value >= 0.5) return 'warning';
    return 'danger';
  }

  function escapeHtml(text) {
    var d = document.createElement('div');
    d.textContent = text || '';
    return d.innerHTML;
  }

  // -----------------------------------------------------------------------
  // Signals Tab
  // -----------------------------------------------------------------------
  function loadSignals() {
    showSpinner('signals-view');
    Promise.all([
      fetchJSON('/api/dashboard/signals'),
      fetchJSON('/api/dashboard/dataset-health').catch(function () { return null; }),
      fetchJSON('/api/dashboard/pipeline-health').catch(function () { return null; }),
    ]).then(function (results) {
      var data = results[0];
      var healthData = results[1];
      var pipelineData = results[2];
      var view = document.getElementById('signals-view');
      if (!data.signals || data.signals.length === 0) {
        showEmpty('signals-view', 'No signals from the latest pipeline run.');
        return;
      }

      var pipelineBanner = '';
      if (pipelineData && pipelineData.pipeline_health) {
        pipelineBanner = renderPipelineHealthBanner(pipelineData.pipeline_health);
      }

      var healthBanner = '';
      if (healthData && healthData.health) {
        healthBanner = renderDatasetHealthBanner(healthData.health);
      }

      var header = '<div class="card" style="margin-bottom:1.25rem">' +
        '<div class="card-header">' +
        '<div><span class="card-title">Latest Signals</span>' +
        '<div class="card-subtitle">' + (data.run_date || '') + '</div></div>' +
        regimeBadge(data.regime || '') +
        '</div></div>';

      var cards = data.signals.map(renderSignalCard).join('');
      view.innerHTML = pipelineBanner + healthBanner + header + '<div class="signals-grid">' + cards + '</div>';
    }).catch(function () {
      showEmpty('signals-view', 'Failed to load signals.');
    });
  }

  function renderDatasetHealthBanner(health) {
    var passed = health.passed;
    var passedCount = health.passed_count || 0;
    var totalChecks = health.total_checks || 0;
    var statusText = passed ? 'PASS' : 'WARN';
    var statusColor = passed ? 'var(--green)' : 'var(--amber, #f59e0b)';
    var bgColor = passed ? 'rgba(34,197,94,0.08)' : 'rgba(245,158,11,0.08)';

    var html = '<div class="card" style="margin-bottom:1.25rem;border-left:3px solid ' + statusColor + ';background:' + bgColor + '">' +
      '<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.5rem">' +
      '<div style="display:flex;align-items:center;gap:0.5rem">' +
      '<span style="font-weight:700;color:' + statusColor + '">' + statusText + '</span>' +
      '<span style="font-size:0.85rem;color:var(--text-secondary)">Dataset Health</span>' +
      '<span style="font-size:0.8rem;color:var(--text-muted)">' + passedCount + '/' + totalChecks + ' checks</span>' +
      '</div>' +
      '<button onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display===\'none\'?\'block\':\'none\'" ' +
      'style="background:none;border:1px solid var(--border);border-radius:4px;padding:2px 8px;cursor:pointer;color:var(--text-secondary);font-size:0.75rem">Details</button>' +
      '<div style="display:none;width:100%;margin-top:0.5rem">';

    var checks = health.checks || [];
    checks.forEach(function (c) {
      var icon = c.passed ? '\u2705' : '\u26A0\uFE0F';
      html += '<div style="font-size:0.8rem;margin:0.25rem 0;color:var(--text-secondary)">' +
        icon + ' <strong>' + escapeHtml(c.name.replace(/_/g, ' ')) + '</strong>: ' +
        escapeHtml(c.detail) + '</div>';
    });

    html += '</div></div></div>';
    return html;
  }

  function renderPipelineHealthBanner(health) {
    var severity = (health.overall_severity || 'pass').toUpperCase();
    var stagesOk = 0;
    var totalStages = 0;
    var stages = health.stages || [];
    stages.forEach(function (s) {
      totalStages++;
      if (s.passed) stagesOk++;
    });

    var colorMap = { PASS: 'var(--green)', WARN: 'var(--amber, #f59e0b)', FAIL: 'var(--red, #ef4444)' };
    var bgMap = { PASS: 'rgba(34,197,94,0.08)', WARN: 'rgba(245,158,11,0.08)', FAIL: 'rgba(239,68,68,0.08)' };
    var statusColor = colorMap[severity] || colorMap.PASS;
    var bgColor = bgMap[severity] || bgMap.PASS;

    var html = '<div class="card" style="margin-bottom:1.25rem;border-left:3px solid ' + statusColor + ';background:' + bgColor + '">' +
      '<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.5rem">' +
      '<div style="display:flex;align-items:center;gap:0.5rem">' +
      '<span style="font-weight:700;color:' + statusColor + '">' + severity + '</span>' +
      '<span style="font-size:0.85rem;color:var(--text-secondary)">Pipeline Health</span>' +
      '<span style="font-size:0.8rem;color:var(--text-muted)">' + stagesOk + '/' + totalStages + ' stages OK</span>' +
      '</div>' +
      '<button onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display===\'none\'?\'block\':\'none\'" ' +
      'style="background:none;border:1px solid var(--border);border-radius:4px;padding:2px 8px;cursor:pointer;color:var(--text-secondary);font-size:0.75rem">Details</button>' +
      '<div style="display:none;width:100%;margin-top:0.5rem">';

    stages.forEach(function (stage) {
      var stageColor = stage.severity === 'pass' ? 'var(--green)' :
                       stage.severity === 'warn' ? 'var(--amber, #f59e0b)' : 'var(--red, #ef4444)';
      var icon = stage.passed ? '\u2705' : (stage.severity === 'fail' ? '\u274C' : '\u26A0\uFE0F');
      html += '<div style="font-size:0.8rem;margin:0.4rem 0;padding:0.3rem 0.5rem;border-left:2px solid ' + stageColor + '">' +
        icon + ' <strong>' + escapeHtml((stage.stage || '').replace(/_/g, ' ')) + '</strong>';

      var checks = stage.checks || [];
      checks.forEach(function (c) {
        if (!c.passed) {
          html += '<div style="font-size:0.75rem;margin-left:1rem;color:var(--text-muted)">\u2022 ' + escapeHtml(c.message) + '</div>';
        }
      });
      html += '</div>';
    });

    var warnings = health.warnings || [];
    if (warnings.length > 0) {
      html += '<div style="margin-top:0.5rem;padding-top:0.5rem;border-top:1px solid var(--border)">' +
        '<div style="font-size:0.75rem;font-weight:600;color:var(--text-secondary)">All Warnings (' + warnings.length + ')</div>';
      warnings.forEach(function (w) {
        html += '<div style="font-size:0.75rem;color:var(--text-muted)">\u2022 ' + escapeHtml(w) + '</div>';
      });
      html += '</div>';
    }

    html += '</div></div></div>';
    return html;
  }

  function renderSignalCard(s) {
    var dirClass = (s.direction || '').toLowerCase() === 'long' ? 'direction-long' : 'direction-short';
    var conf = s.confidence || 0;

    return '<div class="card signal-card">' +
      '<div class="card-header">' +
        '<div>' +
          '<span class="signal-ticker">' + escapeHtml(s.ticker) + '</span>' +
          '<span class="direction-badge ' + dirClass + '">' + escapeHtml(s.direction) + '</span>' +
          '<span class="signal-model">' + escapeHtml(s.signal_model || '') + '</span>' +
        '</div>' +
      '</div>' +
      '<div class="confidence-label">Confidence: ' + fmt(conf, 0) + '/100</div>' +
      '<div class="confidence-meter"><div class="confidence-fill" style="width:' + conf + '%;background:' + confidenceColor(conf) + '"></div></div>' +
      '<div class="trade-grid">' +
        gridItem('Entry', '$' + fmt(s.entry_price)) +
        gridItem('Stop', '$' + fmt(s.stop_loss), 'var(--red)') +
        gridItem('Target', '$' + fmt(s.target_1), 'var(--green)') +
        gridItem('Hold', (s.holding_period_days || '\u2014') + 'd') +
      '</div>' +
      (s.thesis ? '<div class="signal-thesis">' + escapeHtml(s.thesis) + '</div>' : '') +
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
  // Cross-Engine Tab
  // -----------------------------------------------------------------------
  function loadCrossEngine() {
    showSpinner('crossengine-view');

    Promise.all([
      fetchJSON('/api/cross-engine/latest').catch(function () { return null; }),
      fetchJSON('/api/cross-engine/credibility').catch(function () { return null; }),
      fetchJSON('/api/dashboard/dataset-health').catch(function () { return null; }),
    ]).then(function (results) {
      var synthesis = results[0];
      var credibility = results[1];
      var healthData = results[2];
      var view = document.getElementById('crossengine-view');

      if (!synthesis) {
        showEmpty('crossengine-view', 'No cross-engine synthesis data yet. Run the pipeline first.');
        return;
      }

      var html = '';

      // --- Cross-Engine Health Banner ---
      if (healthData && healthData.cross_engine_health) {
        var ceh = healthData.cross_engine_health;
        var cePassed = ceh.passed;
        var ceColor = cePassed ? 'var(--green)' : 'var(--amber, #f59e0b)';
        var ceBg = cePassed ? 'rgba(34,197,94,0.08)' : 'rgba(245,158,11,0.08)';
        html += '<div class="card" style="margin-bottom:1.25rem;border-left:3px solid ' + ceColor + ';background:' + ceBg + '">' +
          '<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.5rem">' +
          '<div style="display:flex;align-items:center;gap:0.5rem">' +
          '<span style="font-weight:700;color:' + ceColor + '">' + (cePassed ? 'PASS' : 'WARN') + '</span>' +
          '<span style="font-size:0.85rem;color:var(--text-secondary)">Cross-Engine Health</span>' +
          '<span style="font-size:0.8rem;color:var(--text-muted)">' + (ceh.passed_count || 0) + '/' + (ceh.total_checks || 0) + ' checks</span>' +
          '</div>' +
          '<button onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display===\'none\'?\'block\':\'none\'" ' +
          'style="background:none;border:1px solid var(--border);border-radius:4px;padding:2px 8px;cursor:pointer;color:var(--text-secondary);font-size:0.75rem">Details</button>' +
          '<div style="display:none;width:100%;margin-top:0.5rem">';
        (ceh.checks || []).forEach(function (c) {
          var icon = c.passed ? '\u2705' : '\u26A0\uFE0F';
          html += '<div style="font-size:0.8rem;margin:0.25rem 0;color:var(--text-secondary)">' +
            icon + ' <strong>' + escapeHtml(c.name.replace(/_/g, ' ')) + '</strong>: ' +
            escapeHtml(c.detail) + '</div>';
        });
        html += '</div></div></div>';
      }

      // --- Header ---
      html += '<div class="card" style="margin-bottom:1.25rem">' +
        '<div class="card-header">' +
        '<div><span class="card-title">Cross-Engine Synthesis</span>' +
        '<div class="card-subtitle">' + (synthesis.run_date || '') + '</div></div>' +
        regimeBadge(synthesis.regime_consensus || '') +
        '</div>' +
        '<div class="metrics-grid" style="margin-bottom:0">' +
        metricCard(synthesis.engines_reporting || 0, 'Engines', null) +
        metricCard(synthesis.convergent_tickers ? Object.keys(synthesis.convergent_tickers).length : 0, 'Convergent', null) +
        metricCard(synthesis.portfolio_recommendation ? synthesis.portfolio_recommendation.length : 0, 'Positions', null) +
        '</div></div>';

      // --- Engine Credibility Cards ---
      if (credibility && credibility.engines && Object.keys(credibility.engines).length > 0) {
        html += '<div class="section-header">' +
          '<div class="section-icon">\uD83C\uDFAF</div>' +
          '<span class="section-title">Engine Credibility</span></div>';

        html += '<div class="engine-grid">';
        Object.keys(credibility.engines).forEach(function (name) {
          var e = credibility.engines[name];
          var hr = e.hit_rate || 0;
          var w = e.weight || 1.0;
          var n = e.resolved_picks || 0;
          var hrPct = Math.round(hr * 100);
          var hrColor = hrPct >= 55 ? 'green' : (hrPct < 45 ? 'red' : 'amber');
          var displayName = name.replace(/_/g, ' ').replace(/\b\w/g, function (c) { return c.toUpperCase(); });

          html += '<div class="engine-card">' +
            '<div class="engine-name">' + escapeHtml(displayName) + '</div>' +
            '<div class="progress-bar"><div class="progress-fill ' + hrColor + '" style="width:' + hrPct + '%"></div></div>' +
            '<div class="engine-stats">' +
              '<div class="engine-stat"><div class="engine-stat-value">' + hrPct + '%</div><div class="engine-stat-label">Hit Rate</div></div>' +
              '<div class="engine-stat"><div class="engine-stat-value">' + fmt(w, 2) + 'x</div><div class="engine-stat-label">Weight</div></div>' +
              '<div class="engine-stat"><div class="engine-stat-value">' + n + '</div><div class="engine-stat-label">Picks</div></div>' +
            '</div></div>';
        });
        html += '</div>';
      }

      // --- Convergent Picks ---
      if (synthesis.convergent_tickers && Object.keys(synthesis.convergent_tickers).length > 0) {
        html += '<div class="section-header">' +
          '<div class="section-icon">\uD83E\uDD1D</div>' +
          '<span class="section-title">Convergent Picks</span></div>';

        html += '<div class="card"><div class="convergent-list">';
        var tickers = synthesis.convergent_tickers;
        Object.keys(tickers).forEach(function (ticker) {
          var engines = tickers[ticker];
          var engineDots = (Array.isArray(engines) ? engines : []).map(function (e) {
            return '<span class="engine-dot">' + escapeHtml(e) + '</span>';
          }).join('');

          html += '<div class="convergent-item">' +
            '<span class="convergent-ticker">' + escapeHtml(ticker) + '</span>' +
            '<div class="convergent-engines">' + engineDots + '</div>' +
            '<span class="convergent-score">' + (Array.isArray(engines) ? engines.length : 0) + ' engines</span>' +
          '</div>';
        });
        html += '</div></div>';
      }

      // --- Portfolio Recommendation ---
      if (synthesis.portfolio_recommendation && synthesis.portfolio_recommendation.length > 0) {
        html += '<div class="section-header">' +
          '<div class="section-icon">\uD83D\uDCBC</div>' +
          '<span class="section-title">Portfolio</span></div>';

        html += '<div class="portfolio-grid">';
        synthesis.portfolio_recommendation.forEach(function (pos) {
          var ticker = pos.ticker || '?';
          var weight = pos.weight_pct || 0;
          var entry = pos.entry_price || 0;
          var stop = pos.stop_loss || 0;
          var target = pos.target_price || 0;
          var hold = pos.holding_period_days || 0;
          var source = pos.source || '';
          var guardianAdj = pos.guardian_adjusted || false;

          var riskPct = entry > 0 && stop > 0 ? Math.abs(entry - stop) / entry * 100 : 0;
          var rewardPct = entry > 0 && target > 0 ? Math.abs(target - entry) / entry * 100 : 0;

          var adjTag = guardianAdj
            ? ' <span class="guardian-adjusted-tag">\u21E9 adj</span>'
            : '';

          html += '<div class="portfolio-card">' +
            '<div class="portfolio-card-header">' +
              '<span class="portfolio-ticker">' + escapeHtml(ticker) + adjTag + '</span>' +
              '<span class="portfolio-weight">' + fmt(weight, 0) + '%</span>' +
            '</div>' +
            '<div class="portfolio-detail">' +
              '<div class="portfolio-detail-item"><div class="portfolio-detail-label">Entry</div><div class="portfolio-detail-value">$' + fmt(entry) + '</div></div>' +
              '<div class="portfolio-detail-item"><div class="portfolio-detail-label">Target</div><div class="portfolio-detail-value positive">+' + fmt(rewardPct, 1) + '%</div></div>' +
              '<div class="portfolio-detail-item"><div class="portfolio-detail-label">Risk</div><div class="portfolio-detail-value negative">' + fmt(riskPct, 1) + '%</div></div>' +
            '</div>' +
            '<div style="margin-top:0.5rem;font-size:0.7rem;color:var(--text-muted)">' +
              'Stop $' + fmt(stop) + ' \u2022 ' + hold + 'd hold \u2022 ' + escapeHtml(source) +
            '</div>' +
          '</div>';
        });
        html += '</div>';
      }

      // --- Executive Summary ---
      if (synthesis.executive_summary) {
        html += '<div class="card" style="margin-top:1rem">' +
          '<div class="card-title" style="margin-bottom:0.5rem">Executive Summary</div>' +
          '<p style="font-size:0.85rem;line-height:1.7;color:var(--text-secondary)">' +
          escapeHtml(synthesis.executive_summary) + '</p></div>';
      }

      // --- Verifier Notes ---
      if (synthesis.verifier_notes) {
        var notes = synthesis.verifier_notes;
        var notesHtml = '';
        if (typeof notes === 'object' && notes !== null) {
          Object.keys(notes).forEach(function (key) {
            var val = notes[key];
            var label = key.replace(/_/g, ' ').replace(/\b\w/g, function (c) { return c.toUpperCase(); });
            if (Array.isArray(val)) {
              notesHtml += '<div style="margin-bottom:0.5rem"><strong>' + escapeHtml(label) + ':</strong></div><ul style="margin:0 0 0.5rem 1rem;padding:0">';
              val.forEach(function (item) {
                notesHtml += '<li>' + escapeHtml(typeof item === 'object' ? JSON.stringify(item) : String(item)) + '</li>';
              });
              notesHtml += '</ul>';
            } else if (typeof val === 'object' && val !== null) {
              notesHtml += '<div style="margin-bottom:0.5rem"><strong>' + escapeHtml(label) + ':</strong></div><ul style="margin:0 0 0.5rem 1rem;padding:0">';
              Object.keys(val).forEach(function (k) {
                var v = val[k];
                var subLabel = k.replace(/_/g, ' ').replace(/\b\w/g, function (c) { return c.toUpperCase(); });
                notesHtml += '<li>' + escapeHtml(subLabel) + ': ' + escapeHtml(typeof v === 'object' ? JSON.stringify(v) : String(v)) + '</li>';
              });
              notesHtml += '</ul>';
            } else {
              notesHtml += '<div style="margin-bottom:0.25rem"><strong>' + escapeHtml(label) + ':</strong> ' + escapeHtml(String(val)) + '</div>';
            }
          });
        } else {
          notesHtml = '<p>' + escapeHtml(String(notes)) + '</p>';
        }
        html += '<div class="card">' +
          '<div class="card-title" style="margin-bottom:0.5rem">Verifier Notes</div>' +
          '<div style="font-size:0.85rem;line-height:1.7;color:var(--text-secondary)">' +
          notesHtml + '</div></div>';
      }

      view.innerHTML = html;
    }).catch(function () {
      showEmpty('crossengine-view', 'Failed to load cross-engine data.');
    });
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
        metricCard(data.total_signals, 'Total Trades', null) +
        metricCard(fmtPct(overall.win_rate * 100), 'Win Rate', overall.win_rate >= 0.5) +
        metricCard(fmtPct(overall.avg_pnl), 'Avg P&L', overall.avg_pnl > 0) +
        metricCard(fmt(risk.sharpe_ratio), 'Sharpe', risk.sharpe_ratio > 0) +
        metricCard(fmt(risk.sortino_ratio), 'Sortino', risk.sortino_ratio > 0) +
        metricCard(fmtPct(risk.max_drawdown_pct), 'Max Drawdown', false) +
        metricCard(fmt(risk.profit_factor), 'Profit Factor', risk.profit_factor > 1) +
        metricCard(fmt(risk.expectancy), 'Expectancy', risk.expectancy > 0) +
      '</div>';

      if (data.equity_curve && data.equity_curve.length > 0) {
        html += '<div class="chart-container">' +
          '<div class="chart-title">Cumulative P&L (Equity Curve)</div>' +
          '<div id="equity-chart"></div>' +
        '</div>';
      }

      if (data.by_model && Object.keys(data.by_model).length > 0) {
        html += breakdownTable('By Signal Model', data.by_model);
      }
      if (data.by_regime && Object.keys(data.by_regime).length > 0) {
        html += breakdownTable('By Regime', data.by_regime);
      }

      view.innerHTML = html;

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
    else if (isPositive === false && value !== '\u2014') cls = ' negative';
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
        '<td>' + escapeHtml(key) + '</td>' +
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
    if (!el || typeof LightweightCharts === 'undefined') return;

    if (equityChart) { equityChart.remove(); equityChart = null; }

    var c = chartColors();
    var chart = LightweightCharts.createChart(el, {
      width: el.clientWidth,
      height: 350,
      layout: { background: { color: c.bg }, textColor: c.text },
      grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } },
      rightPriceScale: { borderColor: c.border },
      timeScale: { borderColor: c.border },
    });
    equityChart = chart;

    var series = chart.addLineSeries({ color: c.teal, lineWidth: 2 });
    series.setData(data.map(function (p) { return { time: p.time, value: p.value }; }));
    chart.timeScale().fitContent();

    window.addEventListener('resize', function () {
      if (equityChart && el.clientWidth > 0) equityChart.applyOptions({ width: el.clientWidth });
    });
  }

  // -----------------------------------------------------------------------
  // Charts Tab
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

      if (equity.equity_curve && equity.equity_curve.length > 0) {
        html += '<div class="chart-container">' +
          '<div class="chart-title">Equity Curve (Cumulative Returns)</div>' +
          '<div id="charts-equity"></div></div>';
      } else {
        html += '<div class="card"><div class="empty-state"><p>No equity curve data yet.</p></div></div>';
      }

      if (drawdown.drawdown && drawdown.drawdown.length > 0) {
        html += '<div class="chart-container">' +
          '<div class="chart-title">Drawdown</div>' +
          '<div id="charts-drawdown"></div></div>';
      } else {
        html += '<div class="card"><div class="empty-state"><p>No drawdown data yet.</p></div></div>';
      }

      if (distribution.distribution && Object.keys(distribution.distribution).length > 0) {
        html += '<div class="card">' +
          '<div class="card-title" style="margin-bottom:0.75rem">Return Distribution by Model</div>';
        Object.keys(distribution.distribution).forEach(function (model) {
          var d = distribution.distribution[model];
          var histogram = renderHistogram(d.returns);
          html += '<div style="margin-bottom:1rem">' +
            '<div style="font-weight:600;margin-bottom:0.25rem">' + escapeHtml(model) +
            ' <span style="color:var(--text-muted)">(n=' + d.count + ', mean=' + fmtPct(d.mean) + ', std=' + fmt(d.std) + ')</span></div>' +
            histogram + '</div>';
        });
        html += '</div>';
      } else {
        html += '<div class="card"><div class="empty-state"><p>No return distribution data yet.</p></div></div>';
      }

      if (regimeMatrix.matrix && regimeMatrix.matrix.length > 0) {
        html += renderRegimeMatrix(regimeMatrix.matrix);
      } else {
        html += '<div class="card"><div class="empty-state"><p>No regime matrix data yet.</p></div></div>';
      }

      if (calibration.calibration && calibration.calibration.length > 0) {
        html += renderCalibrationTable(calibration.calibration);
      } else {
        html += '<div class="card"><div class="empty-state"><p>No calibration data yet.</p></div></div>';
      }

      view.innerHTML = html;

      if (equity.equity_curve && equity.equity_curve.length > 0) {
        renderTimeChart('charts-equity', equity.equity_curve, chartColors().teal, 'line');
      }
      if (drawdown.drawdown && drawdown.drawdown.length > 0) {
        renderTimeChart('charts-drawdown', drawdown.drawdown, chartColors().red, 'area');
      }
    }).catch(function () {
      showEmpty('charts-view', 'Failed to load chart data.');
    });
  }

  function renderTimeChart(containerId, data, color, type) {
    var el = document.getElementById(containerId);
    if (!el || typeof LightweightCharts === 'undefined') return;

    var c = chartColors();
    var chart = LightweightCharts.createChart(el, {
      width: el.clientWidth,
      height: 300,
      layout: { background: { color: c.bg }, textColor: c.text },
      grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } },
      rightPriceScale: { borderColor: c.border },
      timeScale: { borderColor: c.border },
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
    if (!returns || returns.length === 0) return '<p style="color:var(--text-muted)">No data</p>';
    var buckets = {};
    var step = 1;
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
        '<div style="background:' + color + ';height:16px;width:' + width + '%;border-radius:4px;min-width:2px"></div>' +
        '<span style="font-size:0.75rem;color:var(--text-muted)">' + count + '</span>' +
      '</div>';
    }).join('');

    return '<div style="max-width:400px">' + rows + '</div>';
  }

  function renderRegimeMatrix(matrix) {
    var models = {};
    var regimes = new Set();
    matrix.forEach(function (m) {
      if (!models[m.model]) models[m.model] = {};
      models[m.model][m.regime] = m;
      regimes.add(m.regime);
    });

    var regimeList = Array.from(regimes).sort();
    var headerCells = regimeList.map(function (r) { return '<th>' + escapeHtml(r) + '</th>'; }).join('');

    var rows = Object.keys(models).map(function (model) {
      var cells = regimeList.map(function (regime) {
        var cell = models[model][regime];
        if (!cell) return '<td style="color:var(--text-muted)">\u2014</td>';
        var wr = cell.win_rate * 100;
        var bg = wr >= 55 ? 'var(--green-soft)' : (wr < 45 ? 'var(--red-soft)' : 'var(--amber-soft)');
        return '<td style="background:' + bg + ';border-radius:6px">' +
          '<div>' + fmtPct(wr) + '</div>' +
          '<div style="font-size:0.7rem;color:var(--text-muted)">' + cell.trades + ' trades</div>' +
        '</td>';
      }).join('');
      return '<tr><td><b>' + escapeHtml(model) + '</b></td>' + cells + '</tr>';
    }).join('');

    return '<div class="card">' +
      '<div class="card-title" style="margin-bottom:0.75rem">Win Rate by Model x Regime</div>' +
      '<table class="data-table">' +
        '<thead><tr><th>Model</th>' + headerCells + '</tr></thead>' +
        '<tbody>' + rows + '</tbody>' +
      '</table></div>';
  }

  function renderCalibrationTable(data) {
    var rows = data.map(function (b) {
      var error = b.calibration_error;
      var errClass = error > 0.15 ? 'negative' : (error < 0.05 ? 'positive' : '');
      return '<tr>' +
        '<td>' + escapeHtml(b.bucket) + '</td>' +
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
      '</table></div>';
  }

  // -----------------------------------------------------------------------
  // Compare Tab
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
        '<div class="card-title" style="margin-bottom:0.5rem">LLM Uplift: Mode Comparison</div>' +
        '<p class="card-subtitle" style="margin-bottom:1rem">Compare quant_only vs hybrid vs agentic_full performance</p>';

      var modeColors = { agentic_full: 'var(--teal-500)', hybrid: '#a855f7', quant_only: 'var(--green)' };

      html += '<div class="metrics-grid">';
      data.comparison.forEach(function (m) {
        var pnlClass = m.avg_pnl > 0 ? 'positive' : (m.avg_pnl < 0 ? 'negative' : '');
        var borderColor = modeColors[m.mode] || 'var(--teal-500)';
        html += '<div class="metric-card" style="border-top:3px solid ' + borderColor + '">' +
          '<div class="metric-value">' + (m.mode || 'unknown').toUpperCase() + '</div>' +
          '<div class="metric-label">' + m.trades + ' trades</div>' +
          '<div class="metric-label ' + pnlClass + '">WR: ' + fmtPct(m.win_rate * 100) +
          ' | Avg: ' + fmtPct(m.avg_pnl) +
          ' | Total: ' + fmtPct(m.total_return) + '</div>' +
        '</div>';
      });
      html += '</div>';

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
        return '<tr>' +
          '<td>' + escapeHtml(r.run_date) + '</td>' +
          '<td>' + regimeBadge(r.regime) + '</td>' +
          '<td>' + (r.universe_size || '\u2014') + '</td>' +
          '<td>' + (r.candidates_scored || '\u2014') + '</td>' +
          '<td>' + (r.pipeline_duration_s != null ? fmt(r.pipeline_duration_s, 1) + 's' : '\u2014') + '</td>' +
        '</tr>';
      }).join('');

      view.innerHTML = '<div class="card">' +
        '<div class="card-title" style="margin-bottom:0.75rem">Pipeline Runs</div>' +
        '<table class="data-table">' +
          '<thead><tr><th>Date</th><th>Regime</th><th>Universe</th><th>Scored</th><th>Duration</th></tr></thead>' +
          '<tbody>' + rows + '</tbody>' +
        '</table></div>' +
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
        metricCard('$' + fmt(costs.total_cost_usd, 4), 'Total Cost (30d)', null) +
        metricCard(numberFormat(costs.total_tokens_in), 'Tokens In', null) +
        metricCard(numberFormat(costs.total_tokens_out), 'Tokens Out', null) +
        metricCard(fmtPct(cache.hit_rate * 100), 'Cache Hit Rate', cache.hit_rate > 0.5) +
        metricCard(cache.hits, 'Cache Hits', null) +
        metricCard(cache.misses, 'Cache Misses', null) +
        metricCard(cache.total_entries, 'Cached Entries', null) +
        metricCard(cache.evictions, 'Evictions', null) +
      '</div>';

      if (costs.by_agent && Object.keys(costs.by_agent).length > 0) {
        var agentRows = Object.keys(costs.by_agent).map(function (agent) {
          return '<tr><td>' + escapeHtml(agent) + '</td><td>$' + fmt(costs.by_agent[agent], 4) + '</td></tr>';
        }).join('');

        html += '<div class="card">' +
          '<div class="card-title" style="margin-bottom:0.75rem">Cost by Agent</div>' +
          '<table class="data-table">' +
            '<thead><tr><th>Agent</th><th>Cost (USD)</th></tr></thead>' +
            '<tbody>' + agentRows + '</tbody>' +
          '</table></div>';
      }

      if (costs.by_date && Object.keys(costs.by_date).length > 0) {
        var dateRows = Object.keys(costs.by_date).sort().reverse().map(function (d) {
          return '<tr><td>' + escapeHtml(d) + '</td><td>$' + fmt(costs.by_date[d], 4) + '</td></tr>';
        }).join('');

        html += '<div class="card">' +
          '<div class="card-title" style="margin-bottom:0.75rem">Cost by Date</div>' +
          '<table class="data-table">' +
            '<thead><tr><th>Date</th><th>Cost (USD)</th></tr></thead>' +
            '<tbody>' + dateRows + '</tbody>' +
          '</table></div>';
      }

      view.innerHTML = html;
    }).catch(function () {
      showEmpty('costs-view', 'Failed to load cost data.');
    });
  }

  function numberFormat(n) {
    if (n == null) return '\u2014';
    return Number(n).toLocaleString();
  }

  // -----------------------------------------------------------------------
  // Boot: load default tab
  // -----------------------------------------------------------------------
  switchTab('signals');

})();
