/* Dashboard SPA — vanilla JS with tab routing and lazy data loading
   Tabs: Overview | Signals | Engines | Performance | Tracks | System */

(function () {
  'use strict';

  // -----------------------------------------------------------------------
  // Engine display names — avoids naive title-casing of acronyms
  // -----------------------------------------------------------------------
  var ENGINE_DISPLAY = { gemini_stst: 'Gemini STST', koocore_d: 'KooCore-D', mas_quant_screener: 'MAS-Quant-Screener', top3_7d: 'Top3-7D' };
  function engineDisplayName(name) {
    return ENGINE_DISPLAY[name] || name.replace(/_/g, ' ').replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  }

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
    overview: false, signals: false, crossengine: false, performance: false,
    tracks: false, system: false
  };
  var equityChart = null;
  var featureFlags = { cross_engine_enabled: true };  // default true until fetched

  // Fetch feature flags on page load and hide engine UI if disabled
  fetchJSON('/api/config/features').then(function (flags) {
    featureFlags = flags;
    if (!flags.cross_engine_enabled) {
      // Hide Engines tab button
      var engineTab = document.querySelector('[data-tab="crossengine"]');
      if (engineTab) engineTab.style.display = 'none';
    }
  }).catch(function () { /* keep defaults */ });

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
      case 'overview': loadOverview(); break;
      case 'signals': loadSignals(); break;
      case 'crossengine': loadCrossEngine(); break;
      case 'performance': loadPerformance(); break;
      case 'tracks': loadTracks(); break;
      case 'system': loadSystem(); break;
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

  function metricCard(value, label, isPositive) {
    var cls = '';
    if (isPositive === true) cls = ' positive';
    else if (isPositive === false && value !== '\u2014') cls = ' negative';
    return '<div class="metric-card">' +
      '<div class="metric-value' + cls + '">' + value + '</div>' +
      '<div class="metric-label">' + label + '</div>' +
    '</div>';
  }

  function numberFormat(n) {
    if (n == null) return '\u2014';
    return Number(n).toLocaleString();
  }

  // -----------------------------------------------------------------------
  // Sub-tab helper
  // -----------------------------------------------------------------------
  function initSubTabs(viewId) {
    var view = document.getElementById(viewId);
    if (!view) return;
    view.querySelectorAll('.subtab-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var target = btn.dataset.subtab;
        // Toggle active button
        view.querySelectorAll('.subtab-btn').forEach(function (b) { b.classList.remove('subtab-active'); });
        btn.classList.add('subtab-active');
        // Toggle content
        view.querySelectorAll('.subtab-content').forEach(function (c) { c.classList.remove('subtab-visible'); });
        var content = view.querySelector('[data-subtab-content="' + target + '"]');
        if (content) content.classList.add('subtab-visible');
      });
    });
  }

  // =======================================================================
  // OVERVIEW TAB
  // =======================================================================
  function loadOverview() {
    showSpinner('overview-view');

    Promise.all([
      fetchJSON('/api/dashboard/overview').catch(function () { return {}; }),
      fetchJSON('/api/dashboard/signals').catch(function () { return { signals: [] }; }),
      fetchJSON('/api/dashboard/engine-reliability').catch(function () { return { engines: [] }; }),
      fetchJSON('/api/dashboard/performance').catch(function () { return {}; }),
    ]).then(function (results) {
      var ov = results[0];
      var sigData = results[1];
      var relData = results[2];
      var perfData = results[3];
      var view = document.getElementById('overview-view');
      var html = '';

      // Hero row: title + profile badge + regime
      html += '<div class="overview-hero">';
      html += '<div class="overview-hero-left">';
      html += '<span class="overview-hero-title">Dashboard</span>';
      if (ov.profile) {
        html += '<span class="overview-profile-badge">' + escapeHtml(ov.profile) + '</span>';
      }
      if (ov.regime) {
        html += regimeBadge(ov.regime);
      }
      html += '</div>';

      // Status row: last run + mode + trading mode
      html += '<div class="overview-status-row">';
      if (ov.run_date) {
        var dotColor = ov.pipeline_duration_s ? 'dot-green' : 'dot-amber';
        html += '<span class="overview-status-dot ' + dotColor + '"></span>';
        html += '<span style="font-size:0.8rem;color:var(--text-secondary)">Last run: ' + escapeHtml(ov.run_date);
        if (ov.pipeline_duration_s) html += ' (' + fmt(ov.pipeline_duration_s, 0) + 's)';
        html += '</span>';
      }
      if (ov.trading_mode) {
        var modeColor = ov.trading_mode === 'LIVE' ? 'var(--green)' : 'var(--amber)';
        html += '<span style="font-size:0.72rem;font-weight:600;padding:0.15rem 0.5rem;border-radius:12px;border:1px solid ' + modeColor + ';color:' + modeColor + '">' + escapeHtml(ov.trading_mode) + '</span>';
      }
      if (ov.score_tiered_stops) {
        html += '<span style="font-size:0.68rem;color:var(--text-muted);border:1px solid var(--card-border);border-radius:12px;padding:0.12rem 0.45rem">tiered stops</span>';
      }
      html += '</div></div>';

      // Key metrics row
      html += '<div class="metrics-grid">';
      html += metricCard(ov.total_trades || 0, 'Trades (90d)', null);
      html += metricCard(ov.win_rate != null ? fmtPct(ov.win_rate * 100) : '\u2014', 'Win Rate', ov.win_rate >= 0.5);
      html += metricCard(ov.avg_pnl != null ? fmtPct(ov.avg_pnl) : '\u2014', 'Avg P&L', ov.avg_pnl > 0);
      html += metricCard(ov.sharpe != null ? fmt(ov.sharpe) : '\u2014', 'Sharpe', ov.sharpe > 0);
      html += metricCard(ov.profit_factor != null ? fmt(ov.profit_factor) : '\u2014', 'Profit Factor', ov.profit_factor > 1);
      html += metricCard(ov.max_drawdown_pct != null ? '-' + fmtPct(Math.abs(ov.max_drawdown_pct)) : '\u2014', 'Max DD', false);
      html += '</div>';

      // Quick-action cards
      html += '<div class="overview-quick-grid">';

      // Signals card
      var sigCount = (sigData.signals || []).length;
      var sigApproved = ov.signals_approved || sigCount;
      html += '<div class="overview-quick-card" onclick="switchTab(\'signals\')">';
      html += '<div class="overview-quick-card-header"><span class="overview-quick-card-title">Today\'s Signals</span><span class="overview-quick-card-arrow">&rarr;</span></div>';
      html += '<div class="overview-quick-card-value">' + sigApproved + '</div>';
      html += '<div class="overview-quick-card-detail">' + (ov.signals_total || 0) + ' total scored, ' + sigApproved + ' approved</div>';
      if (sigData.signals && sigData.signals.length > 0) {
        html += '<div class="overview-signals-list" style="margin-top:0.75rem">';
        sigData.signals.slice(0, 5).forEach(function (s) {
          var dirClass = (s.direction || '').toLowerCase() === 'long' ? 'positive' : 'negative';
          html += '<div class="overview-signal-row">';
          html += '<span class="overview-signal-ticker">' + escapeHtml(s.ticker) + '</span>';
          html += '<span class="' + dirClass + '" style="font-size:0.75rem;font-weight:600">' + escapeHtml(s.direction) + '</span>';
          html += '<span style="font-size:0.75rem;color:var(--text-muted)">' + escapeHtml(s.signal_model || '') + '</span>';
          html += '<span class="overview-signal-conf">' + fmt(s.confidence, 0) + '</span>';
          html += '</div>';
        });
        if (sigData.signals.length > 5) {
          html += '<div style="text-align:center;font-size:0.75rem;color:var(--text-muted);padding:0.3rem">+' + (sigData.signals.length - 5) + ' more</div>';
        }
        html += '</div>';
      }
      html += '</div>';

      // Engine health card (hidden when cross_engine_enabled=false)
      if (featureFlags.cross_engine_enabled) {
        var engines = relData.engines || [];
        html += '<div class="overview-quick-card" onclick="switchTab(\'crossengine\')">';
        html += '<div class="overview-quick-card-header"><span class="overview-quick-card-title">Engine Health</span><span class="overview-quick-card-arrow">&rarr;</span></div>';
        var healthyCount = engines.filter(function (e) { return e.latest_status === 'success'; }).length;
        html += '<div class="overview-quick-card-value">' + healthyCount + '/' + engines.length + '</div>';
        html += '<div class="overview-quick-card-detail">engines reporting</div>';
        if (engines.length > 0) {
          html += '<div style="margin-top:0.75rem;display:flex;flex-direction:column;gap:0.35rem">';
          engines.forEach(function (e) {
            var status = (e.latest_status || 'no_data').toLowerCase();
            var dotClass = status === 'success' ? 'dot-green' : status === 'no_data' ? 'dot-amber' : 'dot-red';
            var sr7 = e.success_rate_7d != null ? (e.success_rate_7d * 100).toFixed(0) + '%' : '--';
            html += '<div style="display:flex;align-items:center;gap:0.5rem;font-size:0.8rem">';
            html += '<span class="overview-status-dot ' + dotClass + '"></span>';
            html += '<span style="font-weight:600;min-width:100px">' + escapeHtml(engineDisplayName(e.engine_name || '')) + '</span>';
            html += '<span style="color:var(--text-muted)">' + sr7 + ' (7d)</span>';
            html += '</div>';
          });
          html += '</div>';
        }
        html += '</div>';
      }

      // Performance card
      html += '<div class="overview-quick-card" onclick="switchTab(\'performance\')">';
      html += '<div class="overview-quick-card-header"><span class="overview-quick-card-title">Performance</span><span class="overview-quick-card-arrow">&rarr;</span></div>';
      var sharpeStr = ov.sharpe != null ? fmt(ov.sharpe) : '--';
      html += '<div class="overview-quick-card-value' + (ov.sharpe > 0 ? ' positive' : '') + '">' + sharpeStr + '</div>';
      html += '<div class="overview-quick-card-detail">Sharpe ratio (90d)</div>';
      html += '<div style="margin-top:0.75rem;display:grid;grid-template-columns:repeat(3,1fr);gap:0.5rem;text-align:center">';
      html += '<div><div style="font-size:0.7rem;color:var(--text-muted);text-transform:uppercase">Sortino</div><div style="font-weight:700">' + (ov.sortino != null ? fmt(ov.sortino) : '--') + '</div></div>';
      html += '<div><div style="font-size:0.7rem;color:var(--text-muted);text-transform:uppercase">Expectancy</div><div style="font-weight:700">' + (ov.expectancy != null ? fmt(ov.expectancy) : '--') + '</div></div>';
      html += '<div><div style="font-size:0.7rem;color:var(--text-muted);text-transform:uppercase">PF</div><div style="font-weight:700">' + (ov.profit_factor != null ? fmt(ov.profit_factor) : '--') + '</div></div>';
      html += '</div>';
      html += '</div>';

      // Model Scorecard card
      if (perfData && perfData.by_model && Object.keys(perfData.by_model).length > 0) {
        html += '<div class="overview-quick-card" onclick="switchTab(\'performance\')">';
        html += '<div class="overview-quick-card-header"><span class="overview-quick-card-title">Model Scorecard</span><span class="overview-quick-card-arrow">&rarr;</span></div>';
        html += '<div style="margin-top:0.25rem;display:flex;flex-direction:column;gap:0.5rem">';
        var models = Object.keys(perfData.by_model);
        models.forEach(function (model) {
          var m = perfData.by_model[model];
          var wrClass = m.win_rate >= 0.5 ? 'positive' : 'negative';
          var pnlClass = m.avg_pnl > 0 ? 'positive' : (m.avg_pnl < 0 ? 'negative' : '');
          html += '<div style="display:flex;justify-content:space-between;align-items:center;padding:0.4rem 0;border-bottom:1px solid var(--border)">';
          html += '<span style="font-weight:700;font-size:0.85rem">' + escapeHtml(model) + '</span>';
          html += '<span style="font-size:0.8rem">' + m.trades + ' trades</span>';
          html += '<span class="' + wrClass + '" style="font-size:0.8rem;font-weight:600">' + fmtPct(m.win_rate * 100) + ' WR</span>';
          html += '<span class="' + pnlClass + '" style="font-size:0.8rem;font-weight:600">' + fmtPct(m.avg_pnl) + '</span>';
          html += '</div>';
        });
        // Show sniper status if not in by_model
        if (!perfData.by_model['sniper']) {
          html += '<div style="display:flex;justify-content:space-between;align-items:center;padding:0.4rem 0;color:var(--text-muted);font-size:0.8rem">';
          html += '<span style="font-weight:700">sniper</span>';
          html += '<span>waiting for bull/choppy regime</span>';
          html += '</div>';
        }
        html += '</div>';
        html += '</div>';
      }

      html += '</div>'; // close quick-grid

      view.innerHTML = html;
    }).catch(function () {
      showEmpty('overview-view', 'Failed to load overview data.');
    });
  }

  // Make switchTab globally accessible for onclick handlers
  window.switchTab = switchTab;

  // =======================================================================
  // SIGNALS TAB
  // =======================================================================
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

      var pipelineBanner = '';
      if (pipelineData && pipelineData.pipeline_health) {
        pipelineBanner = renderPipelineHealthBanner(pipelineData.pipeline_health);
      }

      var fmpBadge = '';
      if (pipelineData && pipelineData.pipeline_health) {
        fmpBadge = renderFmpEndpointBadge(pipelineData.pipeline_health);
      }

      var healthBanner = '';
      if (healthData && healthData.health) {
        healthBanner = renderDatasetHealthBanner(healthData.health);
      }

      var checklistBanner = '';
      if ((pipelineData && pipelineData.pipeline_health) || (healthData && healthData.health)) {
        checklistBanner = renderVerificationChecklist(
          pipelineData ? pipelineData.pipeline_health : null,
          healthData ? healthData.health : null
        );
      }

      var header = '<div class="card" style="margin-bottom:1.25rem">' +
        '<div class="card-header">' +
        '<div><span class="card-title">Latest Signals</span>' +
        '<div class="card-subtitle">' + (data.run_date || '') + '</div></div>' +
        regimeBadge(data.regime || '') +
        '</div></div>';

      if (!data.signals || data.signals.length === 0) {
        var reason = data.empty_reason || 'No signals from the latest pipeline run.';
        var metaText = '';
        if (data.meta && typeof data.meta.total_signals === 'number') {
          metaText = '<p>Total: ' + data.meta.total_signals +
            ' | Approved: ' + (data.meta.approved_signals || 0) + '</p>';
        }
        view.innerHTML = pipelineBanner + fmpBadge + healthBanner + checklistBanner + header +
          '<div class="empty-state"><h3>No Signals</h3><p>' + escapeHtml(reason) + '</p>' + metaText + '</div>';
        return;
      }

      var cards = data.signals.map(renderSignalCard).join('');
      view.innerHTML = pipelineBanner + fmpBadge + healthBanner + checklistBanner + header + '<div class="signals-grid">' + cards + '</div>';
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
    var problemChecks = checks.filter(function (c) { return !c.passed; });
    var passChecks = checks.filter(function (c) { return c.passed; });
    if (problemChecks.length === 0) {
      html += '<div style="font-size:0.8rem;color:var(--text-secondary)">All ' + passChecks.length + ' dataset checks passed.</div>';
    } else {
      problemChecks.forEach(function (c) {
        html += '<div style="font-size:0.8rem;margin:0.25rem 0;color:var(--text-secondary)">' +
          '\u26A0\uFE0F <strong>' + escapeHtml(c.name.replace(/_/g, ' ')) + '</strong>: ' +
          escapeHtml(c.detail) + '</div>';
      });
    }
    if (passChecks.length > 0) {
      html += '<details style="margin-top:0.45rem"><summary style="cursor:pointer;font-size:0.75rem;color:var(--text-secondary)">Show passing checks (' + passChecks.length + ')</summary>';
      passChecks.forEach(function (c) {
        html += '<div style="font-size:0.75rem;margin:0.22rem 0;color:var(--text-muted)">' +
          '\u2705 <strong>' + escapeHtml(c.name.replace(/_/g, ' ')) + '</strong>: ' +
          escapeHtml(c.detail) + '</div>';
      });
      html += '</details>';
    }

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
      var checks = stage.checks || [];
      var stageProblems = checks.filter(function (c) { return !c.passed; });
      var stagePasses = checks.length - stageProblems.length;
      html += '<div style="font-size:0.8rem;margin:0.35rem 0;padding:0.25rem 0.45rem;border-left:2px solid ' + stageColor + '">' +
        icon + ' <strong>' + escapeHtml((stage.stage || '').replace(/_/g, ' ')) + '</strong>' +
        '<span style="color:var(--text-muted);font-size:0.74rem"> (' + stagePasses + '/' + checks.length + ' checks passed)</span>';

      stageProblems.forEach(function (c) {
        var cSeverity = (c.severity || 'pass').toLowerCase();
        var cColor = cSeverity === 'fail' ? 'var(--red, #ef4444)' :
                     cSeverity === 'warn' ? 'var(--amber, #f59e0b)' : 'var(--green)';
        var cIcon = c.passed ? '\u2705' : (cSeverity === 'fail' ? '\u274C' : '\u26A0\uFE0F');
        html += '<div style="font-size:0.75rem;margin-left:1rem;color:var(--text-muted)">' +
          cIcon + ' <span style="color:' + cColor + '">' + escapeHtml((c.name || '').replace(/_/g, ' ')) + '</span>: ' +
          escapeHtml(c.message || '') + '</div>';
      });
      html += '</div>';
    });

    html += '</div></div></div>';
    return html;
  }

  function renderFmpEndpointBadge(pipelineHealth) {
    if (!pipelineHealth || !pipelineHealth.stages) return '';
    var endpointCheck = null;
    (pipelineHealth.stages || []).forEach(function (stage) {
      if (endpointCheck) return;
      (stage.checks || []).forEach(function (c) {
        if (!endpointCheck && c.name === 'fmp_endpoint_availability') endpointCheck = c;
      });
    });
    if (!endpointCheck) return '';
    var statusMap = endpointCheck.value && endpointCheck.value.endpoints ? endpointCheck.value.endpoints : {};
    if (Object.keys(statusMap).length === 0) return '';

    function badge(status) {
      if (status === 'supported') return { color: 'var(--green)', label: 'supported' };
      if (status === 'per_ticker_only') return { color: 'var(--amber, #f59e0b)', label: 'per-ticker-only' };
      if (status === 'plan_gated') return { color: 'var(--amber, #f59e0b)', label: 'plan-gated' };
      if (status === 'unsupported') return { color: 'var(--amber, #f59e0b)', label: 'unsupported' };
      if (status === 'disabled') return { color: 'var(--red, #ef4444)', label: 'disabled' };
      if (status === 'auth_error') return { color: 'var(--red, #ef4444)', label: 'auth-error' };
      return { color: 'var(--text-muted)', label: status || 'unknown' };
    }

    var html = '<div class="card" style="margin-bottom:1.25rem">' +
      '<div class="card-header"><div><span class="card-title">FMP Endpoint Availability</span></div></div>';
    var callsUsed = endpointCheck.value.calls_used;
    var dailyBudget = endpointCheck.value.daily_budget;
    if (callsUsed != null && dailyBudget != null) {
      html += '<div style="font-size:0.74rem;color:var(--text-muted);margin-bottom:0.45rem">Calls: ' + callsUsed + '/' + dailyBudget + '</div>';
    }
    html += '<div style="display:flex;flex-wrap:wrap;gap:0.4rem">';
    Object.keys(statusMap).forEach(function (name) {
      var meta = badge(statusMap[name]);
      html += '<span style="font-size:0.72rem;border:1px solid ' + meta.color + ';color:' + meta.color +
        ';border-radius:999px;padding:0.12rem 0.42rem">' +
        escapeHtml(name.replace(/_/g, ' ')) + ': ' + escapeHtml(meta.label) + '</span>';
    });
    html += '</div></div>';
    return html;
  }

  function renderVerificationChecklist(pipelineHealth, datasetHealth) {
    var stageItems = [], checkItems = [], datasetItems = [], warningItems = [];

    function normalizeStatus(passed, severity) {
      if (passed) return 'pass';
      if ((severity || '').toLowerCase() === 'fail') return 'fail';
      return 'warn';
    }

    function statusView(status) {
      if (status === 'pass') return { icon: '\u2705', color: 'var(--green)', label: 'PASS' };
      if (status === 'fail') return { icon: '\u274C', color: 'var(--red, #ef4444)', label: 'FAIL' };
      return { icon: '\u26A0\uFE0F', color: 'var(--amber, #f59e0b)', label: 'WARN' };
    }

    function renderRows(items) {
      if (!items.length) return '<div style="font-size:0.78rem;color:var(--text-muted);padding:0.3rem 0">No checks.</div>';
      return items.map(function (item) {
        var v = statusView(item.status);
        return '<div style="padding:0.42rem 0.52rem;border:1px solid var(--border);border-radius:8px;margin-top:0.35rem">' +
          '<div style="display:flex;align-items:center;justify-content:space-between;gap:0.5rem">' +
          '<div style="display:flex;align-items:center;gap:0.35rem;font-size:0.78rem;color:var(--text-secondary)">' +
          '<span style="color:' + v.color + '">' + v.icon + '</span>' +
          '<strong>' + escapeHtml(item.label) + '</strong></div>' +
          '<span style="font-size:0.68rem;color:' + v.color + ';border:1px solid ' + v.color + ';border-radius:999px;padding:0.05rem 0.35rem">' + v.label + '</span></div>' +
          '<div style="font-size:0.74rem;color:var(--text-muted);margin-top:0.18rem">' + escapeHtml(item.detail || '') + '</div></div>';
      }).join('');
    }

    function renderSection(title, items, opts) {
      opts = opts || {};
      var problems = items.filter(function (i) { return i.status !== 'pass'; });
      var passes = items.filter(function (i) { return i.status === 'pass'; });
      var open = problems.length > 0 || !!opts.forceOpen;
      var suffix = problems.length === 0 && items.length > 0 ? ' \u00B7 all pass' :
                   problems.length > 0 ? ' \u00B7 ' + problems.length + ' issues' : '';
      var html = '<details' + (open ? ' open' : '') + (opts.marginTop ? ' style="margin-top:' + opts.marginTop + '"' : '') + '>' +
        '<summary style="cursor:pointer;font-size:0.8rem;color:var(--text-secondary);font-weight:600">' +
        title + ' (' + items.length + ')' + suffix + '</summary>';
      if (problems.length === 0 && items.length > 0) {
        html += '<div style="font-size:0.78rem;color:var(--text-muted);padding:0.35rem 0">All ' + passes.length + ' checks passed.</div>';
      } else {
        html += renderRows(problems);
      }
      if (passes.length > 0) {
        html += '<details style="margin-top:0.35rem"><summary style="cursor:pointer;font-size:0.75rem;color:var(--text-secondary)">Show passing (' + passes.length + ')</summary>' + renderRows(passes) + '</details>';
      }
      return html + '</details>';
    }

    if (pipelineHealth && pipelineHealth.stages) {
      (pipelineHealth.stages || []).forEach(function (stage) {
        var label = (stage.stage || '').replace(/_/g, ' ');
        stageItems.push({ label: label, status: normalizeStatus(!!stage.passed, stage.severity || 'warn'), detail: stage.passed ? 'passed' : 'review needed' });
        (stage.checks || []).forEach(function (c) {
          checkItems.push({ label: label + ' / ' + (c.name || '').replace(/_/g, ' '), status: normalizeStatus(!!c.passed, c.severity || 'warn'), detail: c.message || '' });
        });
      });
    }
    if (datasetHealth && datasetHealth.checks) {
      (datasetHealth.checks || []).forEach(function (c) {
        datasetItems.push({ label: (c.name || '').replace(/_/g, ' '), status: c.passed ? 'pass' : 'warn', detail: c.detail || '' });
      });
    }
    if (pipelineHealth && pipelineHealth.warnings) {
      (pipelineHealth.warnings || []).forEach(function (w) {
        warningItems.push({ label: 'Pipeline warning', status: 'warn', detail: w || '' });
      });
    }

    var all = stageItems.concat(checkItems).concat(datasetItems).concat(warningItems);
    var passCount = all.filter(function (i) { return i.status === 'pass'; }).length;
    var warnCount = all.filter(function (i) { return i.status === 'warn'; }).length;
    var failCount = all.filter(function (i) { return i.status === 'fail'; }).length;

    return '<div class="card" style="margin-bottom:1.25rem">' +
      '<div class="card-header"><div><span class="card-title">Verification Checklist</span></div></div>' +
      '<div style="display:flex;flex-wrap:wrap;gap:0.45rem;margin-bottom:0.6rem">' +
      '<span style="font-size:0.72rem;color:var(--text-secondary);border:1px solid var(--border);border-radius:999px;padding:0.12rem 0.45rem">Total: ' + all.length + '</span>' +
      '<span style="font-size:0.72rem;color:var(--green);border:1px solid var(--green);border-radius:999px;padding:0.12rem 0.45rem">PASS: ' + passCount + '</span>' +
      '<span style="font-size:0.72rem;color:var(--amber, #f59e0b);border:1px solid var(--amber, #f59e0b);border-radius:999px;padding:0.12rem 0.45rem">WARN: ' + warnCount + '</span>' +
      '<span style="font-size:0.72rem;color:var(--red, #ef4444);border:1px solid var(--red, #ef4444);border-radius:999px;padding:0.12rem 0.45rem">FAIL: ' + failCount + '</span>' +
      '</div>' +
      renderSection('Pipeline Stages', stageItems, { forceOpen: true }) +
      renderSection('Pipeline Checks', checkItems, { marginTop: '0.45rem' }) +
      renderSection('Dataset Checks', datasetItems, { marginTop: '0.45rem' }) +
      renderSection('Warnings', warningItems, { marginTop: '0.45rem' }) +
      '</div>';
  }

  function renderSignalCard(s) {
    var dirClass = (s.direction || '').toLowerCase() === 'long' ? 'direction-long' : 'direction-short';
    var conf = s.confidence || 0;
    return '<div class="card signal-card">' +
      '<div class="card-header"><div>' +
        '<span class="signal-ticker">' + escapeHtml(s.ticker) + '</span>' +
        '<span class="direction-badge ' + dirClass + '">' + escapeHtml(s.direction) + '</span>' +
        '<span class="signal-model">' + escapeHtml(s.signal_model || '') + '</span>' +
      '</div></div>' +
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
    return '<div class="trade-grid-item"><div class="label">' + label + '</div><div class="value"' + style + '>' + value + '</div></div>';
  }

  // =======================================================================
  // CROSS-ENGINE TAB
  // =======================================================================
  function loadCrossEngine() {
    showSpinner('crossengine-view');

    Promise.all([
      fetchJSON('/api/cross-engine/latest').catch(function () { return null; }),
      fetchJSON('/api/cross-engine/credibility').catch(function () { return null; }),
      fetchJSON('/api/dashboard/dataset-health').catch(function () { return null; }),
      fetchJSON('/api/dashboard/engine-reliability').catch(function () { return null; }),
    ]).then(function (results) {
      var synthesis = results[0];
      var credibility = results[1];
      var healthData = results[2];
      var reliabilityData = results[3];
      var view = document.getElementById('crossengine-view');

      if (!synthesis) {
        showEmpty('crossengine-view', 'No cross-engine synthesis data yet. Run the pipeline first.');
        return;
      }

      var html = '';

      // Cross-Engine Health Banner
      if (healthData && healthData.cross_engine_health) {
        var ceh = healthData.cross_engine_health;
        var ceColor = ceh.passed ? 'var(--green)' : 'var(--amber, #f59e0b)';
        var ceBg = ceh.passed ? 'rgba(34,197,94,0.08)' : 'rgba(245,158,11,0.08)';
        html += '<div class="card" style="margin-bottom:1.25rem;border-left:3px solid ' + ceColor + ';background:' + ceBg + '">' +
          '<div style="display:flex;align-items:center;gap:0.5rem">' +
          '<span style="font-weight:700;color:' + ceColor + '">' + (ceh.passed ? 'PASS' : 'WARN') + '</span>' +
          '<span style="font-size:0.85rem;color:var(--text-secondary)">Cross-Engine Health</span>' +
          '<span style="font-size:0.8rem;color:var(--text-muted)">' + (ceh.passed_count || 0) + '/' + (ceh.total_checks || 0) + ' checks</span>' +
          '</div></div>';
      }

      if (reliabilityData && reliabilityData.engines && reliabilityData.engines.length > 0) {
        html += renderEngineReliabilityTable(reliabilityData);
      }

      // Header
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

      // Engine Credibility Cards
      if (credibility && credibility.engines && Object.keys(credibility.engines).length > 0) {
        html += '<div class="engine-grid">';
        Object.keys(credibility.engines).forEach(function (name) {
          var e = credibility.engines[name];
          var hrPct = Math.round((e.hit_rate || 0) * 100);
          var hrColor = hrPct >= 55 ? 'green' : (hrPct < 45 ? 'red' : 'amber');
          html += '<div class="engine-card">' +
            '<div class="engine-name">' + escapeHtml(engineDisplayName(name)) + '</div>' +
            '<div class="progress-bar"><div class="progress-fill ' + hrColor + '" style="width:' + hrPct + '%"></div></div>' +
            '<div class="engine-stats">' +
              '<div class="engine-stat"><div class="engine-stat-value">' + hrPct + '%</div><div class="engine-stat-label">Hit Rate</div></div>' +
              '<div class="engine-stat"><div class="engine-stat-value">' + fmt(e.weight || 1.0, 2) + 'x</div><div class="engine-stat-label">Weight</div></div>' +
              '<div class="engine-stat"><div class="engine-stat-value">' + (e.resolved_picks || 0) + '</div><div class="engine-stat-label">Picks</div></div>' +
            '</div></div>';
        });
        html += '</div>';
      }

      // Portfolio Recommendation
      if (synthesis.portfolio_recommendation && synthesis.portfolio_recommendation.length > 0) {
        html += '<div class="card-title" style="margin-bottom:0.75rem">Portfolio</div>';
        html += '<div class="portfolio-grid">';
        synthesis.portfolio_recommendation.forEach(function (pos) {
          var ticker = pos.ticker || '?';
          var weight = pos.weight_pct || 0;
          var entry = pos.entry_price || 0;
          var stop = pos.stop_loss || 0;
          var target = pos.target_price || 0;
          var hold = pos.holding_period_days || 0;
          var guardianAdj = pos.guardian_adjusted || false;
          var riskPct = entry > 0 && stop > 0 ? Math.abs(entry - stop) / entry * 100 : 0;
          var rewardPct = entry > 0 && target > 0 ? Math.abs(target - entry) / entry * 100 : 0;
          var adjTag = guardianAdj ? ' <span class="guardian-adjusted-tag">\u21E9 adj</span>' : '';
          var stratTags = pos.strategy_tags || pos.strategies || [];
          var stratHtml = '';
          if (Array.isArray(stratTags) && stratTags.length > 0) {
            stratHtml = '<div class="strategy-tags-row">';
            stratTags.forEach(function (tag) {
              var cls = tag.indexOf('kc_') === 0 ? 'tag-kc' : (tag.indexOf('gem_') === 0 ? 'tag-gem' : 'tag-other');
              stratHtml += '<span class="strategy-tag ' + cls + '">' + escapeHtml(tag) + '</span>';
            });
            stratHtml += '</div>';
          }
          html += '<div class="portfolio-card">' +
            '<div class="portfolio-card-header">' +
              '<span class="portfolio-ticker">' + escapeHtml(ticker) + adjTag + '</span>' +
              '<span class="portfolio-weight">' + fmt(weight, 0) + '%</span>' +
            '</div>' + stratHtml +
            '<div class="portfolio-detail">' +
              '<div class="portfolio-detail-item"><div class="portfolio-detail-label">Entry</div><div class="portfolio-detail-value">$' + fmt(entry) + '</div></div>' +
              '<div class="portfolio-detail-item"><div class="portfolio-detail-label">Target</div><div class="portfolio-detail-value positive">+' + fmt(rewardPct, 1) + '%</div></div>' +
              '<div class="portfolio-detail-item"><div class="portfolio-detail-label">Risk</div><div class="portfolio-detail-value negative">' + fmt(riskPct, 1) + '%</div></div>' +
            '</div>' +
            '<div style="margin-top:0.5rem;font-size:0.7rem;color:var(--text-muted)">Stop $' + fmt(stop) + ' \u2022 ' + hold + 'd hold</div>' +
          '</div>';
        });
        html += '</div>';
      }

      view.innerHTML = html;
    }).catch(function () {
      showEmpty('crossengine-view', 'Failed to load cross-engine data.');
    });
  }

  function renderEngineReliabilityTable(data) {
    var rows = data.engines || [];
    var html = '<div class="card" style="margin-bottom:1.25rem">' +
      '<div class="card-title" style="margin-bottom:0.65rem">Engine Reliability</div>' +
      '<div style="overflow-x:auto"><table class="data-table"><thead><tr>' +
      '<th>Engine</th><th>Status</th><th>Last Success</th><th>Streak</th><th>7d</th><th>30d</th>' +
      '</tr></thead><tbody>';
    rows.forEach(function (r) {
      var status = String(r.latest_status || 'no_data').toLowerCase();
      var statusColor = status === 'success' ? 'var(--green)' : (status === 'no_data' ? 'var(--text-muted)' : 'var(--amber)');
      var sr7 = r.success_rate_7d == null ? '\u2014' : fmtPct(r.success_rate_7d * 100);
      var sr30 = r.success_rate_30d == null ? '\u2014' : fmtPct(r.success_rate_30d * 100);
      html += '<tr>' +
        '<td>' + escapeHtml(engineDisplayName(r.engine_name || '')) + '</td>' +
        '<td><span style="color:' + statusColor + ';font-weight:600">' + escapeHtml(status) + '</span></td>' +
        '<td>' + escapeHtml(r.last_success_date || '\u2014') + '</td>' +
        '<td>' + (r.consecutive_failures == null ? '\u2014' : r.consecutive_failures) + '</td>' +
        '<td>' + sr7 + '</td><td>' + sr30 + '</td></tr>';
    });
    html += '</tbody></table></div></div>';
    return html;
  }

  // =======================================================================
  // PERFORMANCE TAB (merged: Performance + Charts + Compare)
  // =======================================================================
  var enginePerfChart = null;

  function loadPerformance() {
    showSpinner('performance-view');

    Promise.all([
      fetchJSON('/api/dashboard/performance'),
      fetchJSON('/api/dashboard/equity-curve?days=90').catch(function () { return { equity_curve: [] }; }),
      fetchJSON('/api/dashboard/drawdown?days=90').catch(function () { return { drawdown: [] }; }),
      fetchJSON('/api/dashboard/return-distribution?days=90').catch(function () { return { distribution: {} }; }),
      fetchJSON('/api/dashboard/regime-matrix?days=180').catch(function () { return { matrix: [] }; }),
      fetchJSON('/api/dashboard/calibration').catch(function () { return { calibration: [] }; }),
      fetchJSON('/api/dashboard/mode-comparison').catch(function () { return { comparison: [] }; }),
      fetchJSON('/api/dashboard/engine-strategy-performance').catch(function () { return null; }),
    ]).then(function (results) {
      var data = results[0];
      var equity = results[1];
      var drawdown = results[2];
      var distribution = results[3];
      var regimeMatrix = results[4];
      var calibration = results[5];
      var modeData = results[6];
      var engData = results[7];
      var view = document.getElementById('performance-view');

      // Sub-tab bar
      var html = '<div class="subtab-bar">' +
        '<button class="subtab-btn subtab-active" data-subtab="perf-overview">Overview</button>' +
        '<button class="subtab-btn" data-subtab="perf-charts">Charts</button>' +
        '<button class="subtab-btn" data-subtab="perf-compare">Compare</button>' +
        '</div>';

      // --- Sub-tab: Overview ---
      html += '<div class="subtab-content subtab-visible" data-subtab-content="perf-overview">';
      if (!data.total_signals || data.total_signals === 0) {
        html += '<div class="empty-state"><h3>No Data</h3><p>No closed trades yet.</p></div>';
      } else {
        var overall = data.overall || {};
        var risk = data.risk_metrics || {};
        html += '<div class="metrics-grid">' +
          metricCard(data.total_signals, 'Total Trades', null) +
          metricCard(fmtPct(overall.win_rate * 100), 'Win Rate', overall.win_rate >= 0.5) +
          metricCard(fmtPct(overall.avg_pnl), 'Avg P&L', overall.avg_pnl > 0) +
          metricCard(fmt(risk.sharpe_ratio), 'Sharpe', risk.sharpe_ratio > 0) +
          metricCard(fmt(risk.sortino_ratio), 'Sortino', risk.sortino_ratio > 0) +
          metricCard(risk.max_drawdown_pct != null ? '-' + fmtPct(Math.abs(risk.max_drawdown_pct)) : '\u2014', 'Max Drawdown', false) +
          metricCard(fmt(risk.profit_factor), 'Profit Factor', risk.profit_factor > 1) +
          metricCard(fmt(risk.expectancy), 'Expectancy', risk.expectancy > 0) +
        '</div>';

        if (data.equity_curve && data.equity_curve.length > 0) {
          html += '<div class="chart-container"><div class="chart-title">Equity Curve</div><div id="equity-chart"></div></div>';
        }

        if (data.by_model && Object.keys(data.by_model).length > 0) {
          html += breakdownTable('By Signal Model', data.by_model);
        }
        if (data.by_regime && Object.keys(data.by_regime).length > 0) {
          html += breakdownTable('By Regime', data.by_regime);
        }
      }
      html += '</div>';

      // --- Sub-tab: Charts ---
      html += '<div class="subtab-content" data-subtab-content="perf-charts">';
      if (equity.equity_curve && equity.equity_curve.length > 0) {
        html += '<div class="chart-container"><div class="chart-title">Equity Curve (90d)</div><div id="charts-equity"></div></div>';
      }
      if (drawdown.drawdown && drawdown.drawdown.length > 0) {
        html += '<div class="chart-container"><div class="chart-title">Drawdown</div><div id="charts-drawdown"></div></div>';
      }
      if (distribution.distribution && Object.keys(distribution.distribution).length > 0) {
        html += '<div class="card"><div class="card-title" style="margin-bottom:0.75rem">Return Distribution</div>';
        Object.keys(distribution.distribution).forEach(function (model) {
          var d = distribution.distribution[model];
          html += '<div style="margin-bottom:1rem"><div style="font-weight:600;margin-bottom:0.25rem">' + escapeHtml(model) +
            ' <span style="color:var(--text-muted)">(n=' + d.count + ', mean=' + fmtPct(d.mean) + ')</span></div>' +
            renderHistogram(d.returns) + '</div>';
        });
        html += '</div>';
      }
      if (regimeMatrix.matrix && regimeMatrix.matrix.length > 0) {
        html += renderRegimeMatrix(regimeMatrix.matrix);
      }
      if (calibration.calibration && calibration.calibration.length > 0) {
        html += renderCalibrationTable(calibration.calibration);
      }
      html += '</div>';

      // --- Sub-tab: Compare ---
      html += '<div class="subtab-content" data-subtab-content="perf-compare">';
      if (modeData.comparison && modeData.comparison.length > 0) {
        html += '<div class="card"><div class="card-title" style="margin-bottom:0.75rem">Mode Comparison</div>';
        var modeRows = modeData.comparison.map(function (m) {
          var pnlClass = m.avg_pnl > 0 ? 'positive' : (m.avg_pnl < 0 ? 'negative' : '');
          return '<tr><td><b>' + escapeHtml(m.mode || '?') + '</b></td><td>' + m.trades + '</td>' +
            '<td>' + fmtPct(m.win_rate * 100) + '</td><td class="' + pnlClass + '">' + fmtPct(m.avg_pnl) + '</td></tr>';
        }).join('');
        html += '<table class="data-table"><thead><tr><th>Mode</th><th>Trades</th><th>Win Rate</th><th>Avg P&L</th></tr></thead>' +
          '<tbody>' + modeRows + '</tbody></table></div>';
      }
      if (featureFlags.cross_engine_enabled && engData && engData.engines && engData.engines.length > 0) {
        var sorted = engData.engines.slice().sort(function (a, b) { return b.hit_rate - a.hit_rate; });
        html += '<div class="card"><div class="card-title" style="margin-bottom:0.75rem">Engine Leaderboard</div><div class="metrics-grid">';
        sorted.forEach(function (eng, idx) {
          var color = idx === 0 ? 'var(--teal-500)' : 'var(--slate-400)';
          var pnlClass = (eng.avg_return_pct || 0) > 0 ? 'positive' : ((eng.avg_return_pct || 0) < 0 ? 'negative' : '');
          html += '<div class="metric-card" style="border-top:3px solid ' + color + ';text-align:left;padding:1.25rem">' +
            '<div class="metric-value" style="font-size:1.1rem;margin-bottom:0.5rem">' + escapeHtml(engineDisplayName(eng.engine_name)) + '</div>' +
            '<div style="font-size:1.8rem;font-weight:700;margin-bottom:0.25rem">' + fmtPct(eng.hit_rate * 100) + '</div>' +
            '<div style="font-size:0.75rem;color:var(--text-secondary)">Avg <span class="' + pnlClass + '">' + fmtPct(eng.avg_return_pct) + '</span> \u2022 ' + eng.total_picks + ' picks</div></div>';
        });
        html += '</div></div>';

        if (engData.all_strategies && engData.all_strategies.length > 0) {
          var engineNames = sorted.map(function (e) { return e.engine_name; });
          var lookup = {};
          sorted.forEach(function (eng) {
            lookup[eng.engine_name] = {};
            (eng.strategies || []).forEach(function (s) { lookup[eng.engine_name][s.strategy] = s; });
          });
          html += '<div class="card"><div class="card-title" style="margin-bottom:0.75rem">Strategy Matrix</div>' +
            '<div style="overflow-x:auto"><table class="data-table strategy-matrix"><thead><tr><th>Strategy</th>';
          engineNames.forEach(function (eng) { html += '<th>' + escapeHtml(engineDisplayName(eng)) + '</th>'; });
          html += '</tr></thead><tbody>';
          engData.all_strategies.forEach(function (strat) {
            html += '<tr><td><b>' + escapeHtml(strat) + '</b></td>';
            engineNames.forEach(function (eng) {
              var s = lookup[eng] && lookup[eng][strat];
              if (s && s.picks > 0) {
                var cellClass = s.hit_rate >= 0.6 ? 'cell-good' : (s.hit_rate >= 0.4 ? 'cell-warn' : 'cell-poor');
                html += '<td class="' + cellClass + '">' + fmtPct(s.hit_rate * 100) + '<br><span style="font-size:0.65rem;color:var(--text-muted)">' + s.picks + ' picks</span></td>';
              } else {
                html += '<td style="color:var(--text-muted)">\u2014</td>';
              }
            });
            html += '</tr>';
          });
          html += '</tbody></table></div></div>';
        }

        if (engData.time_series && engData.time_series.length > 0) {
          var trendColors = { koocore_d: '#14b8a6', gemini_stst: '#a855f7', top3_7d: '#f59e0b' };
          var trendEngines = {};
          engData.time_series.forEach(function (pt) {
            if (!trendEngines[pt.engine_name]) trendEngines[pt.engine_name] = [];
            trendEngines[pt.engine_name].push(pt);
          });
          var legendHtml = '<div class="chart-legend">';
          Object.keys(trendEngines).sort().forEach(function (name, i) {
            var color = trendColors[name] || ['#14b8a6', '#a855f7', '#f59e0b'][i % 3];
            legendHtml += '<div class="chart-legend-item"><span class="chart-legend-swatch" style="background:' + color + '"></span><span>' + escapeHtml(engineDisplayName(name)) + '</span></div>';
          });
          legendHtml += '</div>';
          html += '<div class="card"><div class="card-title" style="margin-bottom:0.5rem">Performance Trend</div>' + legendHtml +
            '<div id="engine-perf-chart" style="height:300px"></div></div>';
        }
      }
      html += '</div>';

      view.innerHTML = html;
      initSubTabs('performance-view');

      // Render charts
      if (data.equity_curve && data.equity_curve.length > 0) {
        renderEquityCurve(data.equity_curve);
      }
      if (equity.equity_curve && equity.equity_curve.length > 0) {
        renderTimeChart('charts-equity', equity.equity_curve, chartColors().teal, 'line');
      }
      if (drawdown.drawdown && drawdown.drawdown.length > 0) {
        renderTimeChart('charts-drawdown', drawdown.drawdown, chartColors().red, 'area');
      }
      if (featureFlags.cross_engine_enabled && engData && engData.time_series && engData.time_series.length > 0) {
        renderEnginePerfChart(engData.time_series);
      }
    }).catch(function () {
      showEmpty('performance-view', 'Failed to load performance data.');
    });
  }

  function breakdownTable(title, data) {
    var rows = Object.keys(data).map(function (key) {
      var d = data[key];
      var pnlClass = d.avg_pnl > 0 ? 'positive' : (d.avg_pnl < 0 ? 'negative' : '');
      var pfVal = d.profit_factor != null ? fmt(d.profit_factor) : '\u2014';
      var pfClass = d.profit_factor != null && d.profit_factor > 1 ? 'positive' : '';
      return '<tr><td>' + escapeHtml(key) + '</td><td>' + d.trades + '</td><td>' + fmtPct(d.win_rate * 100) + '</td><td class="' + pnlClass + '">' + fmtPct(d.avg_pnl) + '</td><td class="' + pfClass + '">' + pfVal + '</td></tr>';
    }).join('');
    return '<div class="card"><div class="card-title" style="margin-bottom:0.75rem">' + title + '</div>' +
      '<table class="data-table"><thead><tr><th>Name</th><th>Trades</th><th>Win Rate</th><th>Avg P&L</th><th>Profit Factor</th></tr></thead><tbody>' + rows + '</tbody></table></div>';
  }

  function renderEquityCurve(data) {
    var el = document.getElementById('equity-chart');
    if (!el || typeof LightweightCharts === 'undefined') return;
    if (equityChart) { equityChart.remove(); equityChart = null; }
    var c = chartColors();
    var chart = LightweightCharts.createChart(el, {
      width: el.clientWidth, height: 350,
      layout: { background: { color: c.bg }, textColor: c.text },
      grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } },
      rightPriceScale: { borderColor: c.border },
      timeScale: { borderColor: c.border },
    });
    equityChart = chart;
    var series = chart.addLineSeries({ color: c.teal, lineWidth: 2 });
    series.setData(data.map(function (p) { return { time: p.time, value: p.value }; }));
    chart.timeScale().fitContent();
    window.addEventListener('resize', function () { if (equityChart && el.clientWidth > 0) equityChart.applyOptions({ width: el.clientWidth }); });
  }

  function renderTimeChart(containerId, data, color, type) {
    var el = document.getElementById(containerId);
    if (!el || typeof LightweightCharts === 'undefined') return;
    var c = chartColors();
    var chart = LightweightCharts.createChart(el, {
      width: el.clientWidth, height: 300,
      layout: { background: { color: c.bg }, textColor: c.text },
      grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } },
      rightPriceScale: { borderColor: c.border },
      timeScale: { borderColor: c.border },
    });
    var series;
    if (type === 'area') {
      series = chart.addAreaSeries({ lineColor: color, topColor: color + '80', bottomColor: color + '10', lineWidth: 2 });
    } else {
      series = chart.addLineSeries({ color: color, lineWidth: 2 });
    }
    series.setData(data.map(function (p) { return { time: p.time, value: p.value }; }));
    chart.timeScale().fitContent();
    window.addEventListener('resize', function () { if (el.clientWidth > 0) chart.applyOptions({ width: el.clientWidth }); });
  }

  function renderHistogram(returns) {
    if (!returns || returns.length === 0) return '<p style="color:var(--text-muted)">No data</p>';
    var buckets = {};
    returns.forEach(function (r) { var k = Math.floor(r) + '%'; buckets[k] = (buckets[k] || 0) + 1; });
    var maxCount = Math.max.apply(null, Object.values(buckets));
    return '<div style="max-width:400px">' + Object.keys(buckets).sort(function (a, b) { return parseFloat(a) - parseFloat(b); }).map(function (key) {
      var count = buckets[key];
      var width = Math.round(count / maxCount * 100);
      var color = parseFloat(key) >= 0 ? 'var(--green)' : 'var(--red)';
      return '<div style="display:flex;align-items:center;gap:0.5rem;margin:2px 0">' +
        '<span style="min-width:50px;text-align:right;font-size:0.8rem">' + key + '</span>' +
        '<div style="background:' + color + ';height:16px;width:' + width + '%;border-radius:4px;min-width:2px"></div>' +
        '<span style="font-size:0.75rem;color:var(--text-muted)">' + count + '</span></div>';
    }).join('') + '</div>';
  }

  function renderRegimeMatrix(matrix) {
    var models = {};
    var regimes = new Set();
    matrix.forEach(function (m) { if (!models[m.model]) models[m.model] = {}; models[m.model][m.regime] = m; regimes.add(m.regime); });
    var regimeList = Array.from(regimes).sort();
    var headerCells = regimeList.map(function (r) { return '<th>' + escapeHtml(r) + '</th>'; }).join('');
    var rows = Object.keys(models).map(function (model) {
      return '<tr><td><b>' + escapeHtml(model) + '</b></td>' + regimeList.map(function (regime) {
        var cell = models[model][regime];
        if (!cell) return '<td>\u2014</td>';
        var wr = cell.win_rate * 100;
        var bg = wr >= 55 ? 'var(--green-soft)' : (wr < 45 ? 'var(--red-soft)' : 'var(--amber-soft)');
        return '<td style="background:' + bg + ';border-radius:6px">' + fmtPct(wr) + '<div style="font-size:0.7rem;color:var(--text-muted)">' + cell.trades + ' trades</div></td>';
      }).join('') + '</tr>';
    }).join('');
    return '<div class="card"><div class="card-title" style="margin-bottom:0.75rem">Win Rate by Model x Regime</div>' +
      '<table class="data-table"><thead><tr><th>Model</th>' + headerCells + '</tr></thead><tbody>' + rows + '</tbody></table></div>';
  }

  function renderCalibrationTable(data) {
    var rows = data.map(function (b) {
      var errClass = b.calibration_error > 0.15 ? 'negative' : (b.calibration_error < 0.05 ? 'positive' : '');
      return '<tr><td>' + escapeHtml(b.bucket) + '</td><td>' + b.trades + '</td><td>' + fmtPct(b.expected_win_rate * 100) + '</td>' +
        '<td>' + fmtPct(b.actual_win_rate * 100) + '</td><td class="' + errClass + '">' + fmt(b.calibration_error, 4) + '</td></tr>';
    }).join('');
    return '<div class="card"><div class="card-title" style="margin-bottom:0.75rem">Confidence Calibration</div>' +
      '<table class="data-table"><thead><tr><th>Bucket</th><th>Trades</th><th>Expected WR</th><th>Actual WR</th><th>Cal. Error</th></tr></thead><tbody>' + rows + '</tbody></table></div>';
  }

  function renderEnginePerfChart(timeSeries) {
    var el = document.getElementById('engine-perf-chart');
    if (!el || typeof LightweightCharts === 'undefined') return;
    if (enginePerfChart) { enginePerfChart.remove(); enginePerfChart = null; }
    var c = chartColors();
    var chart = LightweightCharts.createChart(el, {
      width: el.clientWidth, height: 300,
      layout: { background: { color: c.bg }, textColor: c.text },
      grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } },
      rightPriceScale: { borderColor: c.border },
      timeScale: { borderColor: c.border },
    });
    enginePerfChart = chart;
    var trendColors = { koocore_d: '#14b8a6', gemini_stst: '#a855f7', top3_7d: '#f59e0b' };
    var byEngine = {};
    timeSeries.forEach(function (pt) { if (!byEngine[pt.engine_name]) byEngine[pt.engine_name] = []; byEngine[pt.engine_name].push(pt); });
    Object.keys(byEngine).sort().forEach(function (name, i) {
      var color = trendColors[name] || ['#14b8a6', '#a855f7', '#f59e0b'][i % 3];
      var series = chart.addLineSeries({ color: color, lineWidth: 2, title: engineDisplayName(name) });
      series.setData(byEngine[name].map(function (p) { return { time: p.week, value: p.cum_return_pct }; }));
    });
    chart.timeScale().fitContent();
    window.addEventListener('resize', function () { if (enginePerfChart && el.clientWidth > 0) enginePerfChart.applyOptions({ width: el.clientWidth }); });
  }

  // =======================================================================
  // TRACKS TAB (merged: Tracks + Backtest)
  // =======================================================================
  var tracksDaysSelect = 14;
  var tracksEquityChart = null;
  var tracksStatusFilter = 'all';
  var tracksSortCol = 'composite_score';
  var tracksSortAsc = false;
  var tracksData = null;
  var backtestChart = null;
  var backtestRuns = [];

  function loadTracks() {
    showSpinner('tracks-view');

    Promise.all([
      fetchJSON('/api/tracks/leaderboard?days=' + tracksDaysSelect).catch(function () { return { tracks: [] }; }),
      fetchJSON('/api/dashboard/backtest/runs').catch(function () { return { runs: [] }; }),
    ]).then(function (results) {
      var trackData = results[0];
      var btData = results[1];
      var view = document.getElementById('tracks-view');

      tracksData = trackData;
      backtestRuns = btData.runs || [];

      // Sub-tab bar
      var html = '<div class="subtab-bar">' +
        '<button class="subtab-btn subtab-active" data-subtab="tracks-leaderboard">Shadow Tracks</button>' +
        '<button class="subtab-btn" data-subtab="tracks-backtest">Backtest</button>' +
        '</div>';

      // --- Sub-tab: Shadow Tracks ---
      html += '<div class="subtab-content subtab-visible" data-subtab-content="tracks-leaderboard">';
      if (!trackData.tracks || trackData.tracks.length === 0) {
        html += '<div class="empty-state"><p>No shadow tracks configured. Enable with SHADOW_TRACKS_ENABLED=true.</p></div>';
      } else {
        html += renderTracksContent(trackData);
      }
      html += '</div>';

      // --- Sub-tab: Backtest ---
      html += '<div class="subtab-content" data-subtab-content="tracks-backtest">';
      if (backtestRuns.length === 0) {
        html += '<div class="empty-state"><p>No backtest results found.</p></div>';
      } else {
        var options = backtestRuns.map(function (r) {
          var range = r.date_range ? (r.date_range.start + ' to ' + r.date_range.end) : '';
          var t = r.total_trades_all_tracks || 0;
          return '<option value="' + escapeHtml(r.filename) + '">' + escapeHtml(range + ' | ' + t + ' trades') + '</option>';
        }).join('');
        html += '<select class="backtest-select" id="backtest-run-select"><option value="">Select a backtest run...</option>' + options + '</select>';
        html += '<div id="backtest-detail"></div>';
      }
      html += '</div>';

      view.innerHTML = html;
      initSubTabs('tracks-view');

      // Bind track events
      if (trackData.tracks && trackData.tracks.length > 0) {
        bindTrackEvents(view, trackData);
      }

      // Bind backtest select
      var btSelect = document.getElementById('backtest-run-select');
      if (btSelect) {
        btSelect.addEventListener('change', function () {
          if (this.value) loadBacktestDetail(this.value);
          else document.getElementById('backtest-detail').innerHTML = '';
        });
      }
    }).catch(function () {
      showEmpty('tracks-view', 'Failed to load tracks data.');
    });
  }

  function renderTracksContent(data) {
    var tracks = filterAndSortTracks(data.tracks);
    var counts = { all: data.tracks.length, active: 0, eliminated: 0, promoted: 0 };
    data.tracks.forEach(function (t) { if (counts[t.status] !== undefined) counts[t.status]++; });

    var html = '<div class="tracks-controls"><div class="tracks-filters">';
    ['all', 'active', 'eliminated', 'promoted'].forEach(function (s) {
      if (s !== 'all' && counts[s] === 0) return;
      var active = tracksStatusFilter === s ? ' tracks-filter-active' : '';
      html += '<button class="tracks-filter-btn' + active + '" data-filter="' + s + '">' +
        s.charAt(0).toUpperCase() + s.slice(1) + ' <span class="tracks-filter-count">' + counts[s] + '</span></button>';
    });
    html += '</div><select id="tracks-days-select" class="tracks-select">' +
      '<option value="7"' + (tracksDaysSelect === 7 ? ' selected' : '') + '>7 days</option>' +
      '<option value="14"' + (tracksDaysSelect === 14 ? ' selected' : '') + '>14 days</option>' +
      '<option value="30"' + (tracksDaysSelect === 30 ? ' selected' : '') + '>30 days</option>' +
      '</select></div>';

    // Leaderboard table
    var cols = [
      { key: 'rank', label: '#' }, { key: 'name', label: 'Track', s: true }, { key: 'status', label: 'Status' },
      { key: 'resolved_picks', label: 'Resolved', s: true }, { key: 'win_rate', label: 'WR', s: true },
      { key: 'avg_return_pct', label: 'Avg Ret', s: true }, { key: 'sharpe_ratio', label: 'Sharpe', s: true },
      { key: 'profit_factor', label: 'PF', s: true }, { key: 'composite_score', label: 'Composite', s: true },
      { key: 'delta_sharpe', label: '\u0394 Sharpe', s: true },
    ];

    html += '<div class="card tracks-table-wrap"><table class="data-table tracks-table"><thead><tr>';
    cols.forEach(function (c) {
      if (c.s) {
        var arrow = tracksSortCol === c.key ? (tracksSortAsc ? ' \u25B2' : ' \u25BC') : '';
        html += '<th class="sortable-th" data-sort="' + c.key + '">' + c.label + arrow + '</th>';
      } else {
        html += '<th>' + c.label + '</th>';
      }
    });
    html += '</tr></thead><tbody>';

    tracks.forEach(function (t, i) {
      var statusBadge = t.status === 'active' ? '<span class="badge badge-success">active</span>' :
        t.status === 'eliminated' ? '<span class="badge badge-failed">elim</span>' :
        t.status === 'promoted' ? '<span class="badge badge-bull">promoted</span>' :
        '<span class="badge badge-unknown">' + t.status + '</span>';
      var rowClass = t.has_sufficient_data ? '' : ' class="tracks-insufficient"';
      var composite = t.composite_score || 0;
      var compositeColor = compositeScoreColor(composite, tracks);

      html += '<tr' + rowClass + '>' +
        '<td>' + (i + 1) + '</td>' +
        '<td><a href="#" class="track-detail-link" data-track="' + t.name + '">' + t.name + '</a></td>' +
        '<td>' + statusBadge + '</td>' +
        '<td>' + t.resolved_picks + '</td>' +
        '<td>' + fmtPctSafe(t.win_rate) + '</td>' +
        '<td class="' + posNegClass(t.avg_return_pct) + '">' + signedPct(t.avg_return_pct) + '</td>' +
        '<td>' + fmtNum(t.sharpe_ratio) + '</td>' +
        '<td>' + fmtNum(t.profit_factor) + '</td>' +
        '<td><span class="tracks-composite" style="background:' + compositeColor + '">' + composite.toFixed(3) + '</span></td>' +
        tracksDeltaCell(t.delta_sharpe) +
        '</tr>';
    });

    html += '</tbody></table></div>';
    html += '<div class="card" style="margin-top:16px"><h3 style="margin-bottom:8px">Equity Curves</h3><div id="tracks-equity-chart" style="height:350px"></div></div>';
    html += '<div id="track-detail-panel" class="card tracks-detail-panel"><div class="tracks-detail-header"><h3 id="track-detail-title">Track Config</h3>' +
      '<button id="track-detail-close" class="tracks-detail-close">&times;</button></div>' +
      '<pre id="track-detail-config" class="tracks-config-pre"></pre><div id="track-detail-picks"></div></div>';

    return html;
  }

  function bindTrackEvents(view, data) {
    var sel = document.getElementById('tracks-days-select');
    if (sel) {
      sel.addEventListener('change', function () {
        tracksDaysSelect = parseInt(sel.value);
        loaded.tracks = false;
        loadTracks();
      });
    }
    view.querySelectorAll('.tracks-filter-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        tracksStatusFilter = btn.dataset.filter;
        // Re-render just tracks content
        var content = view.querySelector('[data-subtab-content="tracks-leaderboard"]');
        if (content) {
          content.innerHTML = renderTracksContent(tracksData);
          bindTrackEvents(view, tracksData);
        }
      });
    });
    view.querySelectorAll('.sortable-th').forEach(function (th) {
      th.addEventListener('click', function () {
        var col = th.dataset.sort;
        if (tracksSortCol === col) tracksSortAsc = !tracksSortAsc;
        else { tracksSortCol = col; tracksSortAsc = false; }
        var content = view.querySelector('[data-subtab-content="tracks-leaderboard"]');
        if (content) {
          content.innerHTML = renderTracksContent(tracksData);
          bindTrackEvents(view, tracksData);
        }
      });
    });
    view.querySelectorAll('.track-detail-link').forEach(function (link) {
      link.addEventListener('click', function (e) {
        e.preventDefault();
        showTrackDetail(link.dataset.track, data.tracks);
      });
    });
    var closeBtn = document.getElementById('track-detail-close');
    if (closeBtn) closeBtn.addEventListener('click', function () { document.getElementById('track-detail-panel').style.display = 'none'; });
    loadTracksEquityCurves(data.tracks);
  }

  function filterAndSortTracks(tracks) {
    var filtered = tracksStatusFilter === 'all' ? tracks : tracks.filter(function (t) { return t.status === tracksStatusFilter; });
    return filtered.slice().sort(function (a, b) {
      var va = a[tracksSortCol], vb = b[tracksSortCol];
      if (typeof va === 'string') return tracksSortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
      va = va || 0; vb = vb || 0;
      return tracksSortAsc ? va - vb : vb - va;
    });
  }

  function compositeScoreColor(score, tracks) {
    var maxScore = 0;
    tracks.forEach(function (t) { if (t.composite_score > maxScore) maxScore = t.composite_score; });
    if (maxScore === 0) return 'rgba(100,100,100,0.15)';
    var pct = score / maxScore;
    if (pct >= 0.7) return 'rgba(34, 197, 94, 0.2)';
    if (pct >= 0.4) return 'rgba(245, 158, 11, 0.15)';
    if (pct > 0) return 'rgba(239, 68, 68, 0.15)';
    return 'rgba(100,100,100,0.1)';
  }

  function tracksDeltaCell(val) {
    if (val == null) return '<td class="tracks-delta">--</td>';
    var cls = val > 0 ? 'positive' : val < 0 ? 'negative' : '';
    return '<td class="tracks-delta ' + cls + '">' + (val > 0 ? '+' : '') + fmtNum(val) + '</td>';
  }

  function posNegClass(val) { return val > 0 ? 'positive' : val < 0 ? 'negative' : ''; }
  function signedPct(val) { return (val > 0 ? '+' : '') + fmtNum(val) + '%'; }
  function fmtNum(v) { return v != null ? v.toFixed(2) : '--'; }
  function fmtPctSafe(v) { return v != null ? (v * 100).toFixed(1) + '%' : '--'; }

  function loadTracksEquityCurves(tracks) {
    var container = document.getElementById('tracks-equity-chart');
    if (!container || typeof LightweightCharts === 'undefined') return;
    if (tracksEquityChart) { tracksEquityChart.remove(); tracksEquityChart = null; }
    var colors = chartColors();
    tracksEquityChart = LightweightCharts.createChart(container, {
      width: container.clientWidth, height: 350,
      layout: { background: { type: 'solid', color: colors.bg }, textColor: colors.text },
      grid: { vertLines: { color: colors.grid }, horzLines: { color: colors.grid } },
      rightPriceScale: { borderColor: colors.border },
      timeScale: { borderColor: colors.border },
    });
    var lineColors = ['#14b8a6', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#22c55e', '#06b6d4'];
    var activeNames = tracks.filter(function (t) { return t.status === 'active'; }).map(function (t) { return t.name; });
    Promise.all(activeNames.map(function (name) {
      return fetchJSON('/api/tracks/' + encodeURIComponent(name) + '/equity').catch(function () { return { snapshots: [] }; });
    })).then(function (results) {
      var hasData = false;
      results.forEach(function (eq, i) {
        if (!eq.snapshots || eq.snapshots.length === 0) return;
        hasData = true;
        var series = tracksEquityChart.addLineSeries({ color: lineColors[i % lineColors.length], lineWidth: 2, title: activeNames[i] });
        series.setData(eq.snapshots.map(function (s) { return { time: s.date, value: s.total_return || 0 }; }));
      });
      if (!hasData) {
        container.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:2rem">No equity data yet.</p>';
      } else {
        tracksEquityChart.timeScale().fitContent();
      }
    });
    new ResizeObserver(function () { if (tracksEquityChart) tracksEquityChart.applyOptions({ width: container.clientWidth }); }).observe(container);
  }

  function showTrackDetail(trackName, allTracks) {
    var panel = document.getElementById('track-detail-panel');
    if (!panel) return;
    panel.style.display = 'block';
    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    var track = allTracks.find(function (t) { return t.name === trackName; });
    document.getElementById('track-detail-title').textContent = trackName + (track ? ' (Gen ' + track.generation + ')' : '');
    document.getElementById('track-detail-config').textContent = track ? JSON.stringify(track.config, null, 2) : '{}';
    var picksEl = document.getElementById('track-detail-picks');
    var descHtml = '';
    if (track && track.description) descHtml = '<p style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:12px">' + track.description + '</p>';
    if (track) {
      descHtml += '<div class="tracks-scorecard-row">' +
        scorecardPill('Win Rate', fmtPctSafe(track.win_rate)) +
        scorecardPill('Sharpe', fmtNum(track.sharpe_ratio)) +
        scorecardPill('PF', fmtNum(track.profit_factor)) +
        scorecardPill('Composite', (track.composite_score || 0).toFixed(3)) +
        '</div>';
    }
    picksEl.innerHTML = descHtml + '<div class="spinner" style="margin-top:12px">Loading picks...</div>';
    fetchJSON('/api/tracks/' + encodeURIComponent(trackName) + '/picks?limit=20').then(function (data) {
      var h = descHtml;
      if (!data.picks || data.picks.length === 0) { picksEl.innerHTML = h + '<p style="color:var(--text-muted)">No picks yet.</p>'; return; }
      h += '<h4 style="margin-top:16px;margin-bottom:8px">Recent Picks</h4>' +
        '<div style="overflow-x:auto"><table class="data-table"><thead><tr><th>Date</th><th>Ticker</th><th>Strategy</th><th>Conf</th><th>Return</th></tr></thead><tbody>';
      data.picks.forEach(function (p) {
        var retClass = p.actual_return > 0 ? 'positive' : p.actual_return < 0 ? 'negative' : '';
        h += '<tr><td>' + p.run_date + '</td><td><strong>' + p.ticker + '</strong></td><td>' + p.strategy + '</td>' +
          '<td>' + (p.confidence || 0).toFixed(1) + '</td><td class="' + retClass + '">' + (p.outcome_resolved ? (p.actual_return > 0 ? '+' : '') + p.actual_return.toFixed(2) + '%' : '--') + '</td></tr>';
      });
      h += '</tbody></table></div>';
      picksEl.innerHTML = h;
    }).catch(function () { picksEl.innerHTML = descHtml + '<p style="color:var(--text-muted)">Failed to load picks.</p>'; });
  }

  function scorecardPill(label, value) {
    return '<div class="tracks-pill"><span class="tracks-pill-label">' + label + '</span><span class="tracks-pill-value">' + value + '</span></div>';
  }

  // Backtest detail (within tracks tab)
  function loadBacktestDetail(filename) {
    var detail = document.getElementById('backtest-detail');
    detail.innerHTML = '<div class="spinner">Loading...</div>';
    fetchJSON('/api/dashboard/backtest/' + encodeURIComponent(filename)).then(function (data) {
      var html = '';
      var range = data.date_range ? (data.date_range.start + ' to ' + data.date_range.end) : '';
      html += '<div class="card" style="margin-bottom:1.25rem"><div class="card-header"><div>' +
        '<span class="card-title">Backtest Results</span><div class="card-subtitle">' + escapeHtml(range) + '</div></div></div>' +
        '<div class="metrics-grid" style="margin-bottom:0">' +
        metricCard(data.trading_days || 0, 'Trading Days', null) +
        metricCard(data.elapsed_s ? fmt(data.elapsed_s, 1) + 's' : '\u2014', 'Elapsed', null) +
        metricCard((data.engines || []).join(', '), 'Engines', null) +
        '</div></div>';

      var synthTrack = (data.synthesis || {}).credibility_weight || (data.synthesis || {}).equal_weight;
      if (synthTrack && synthTrack.summary) {
        var s = synthTrack.summary;
        html += '<div class="metrics-grid">' +
          metricCard(s.total_trades || 0, 'Trades', null) +
          metricCard(fmtPct((s.win_rate || 0) * 100), 'Win Rate', s.win_rate >= 0.5) +
          metricCard(fmt(s.sharpe_ratio), 'Sharpe', s.sharpe_ratio > 0) +
          metricCard(fmt(s.profit_factor), 'PF', s.profit_factor > 1) +
          metricCard(fmt(s.expectancy), 'Expectancy', s.expectancy > 0) +
          metricCard(s.max_drawdown_pct != null ? '-' + fmtPct(Math.abs(s.max_drawdown_pct)) : '\u2014', 'Max DD', false) +
          '</div>';
      }

      var perEngine = data.per_engine || {};
      var engineNames = Object.keys(perEngine);
      if (engineNames.length > 0) {
        html += '<div class="card"><div class="card-title" style="margin-bottom:0.75rem">Per-Engine Summary</div>' +
          '<div style="overflow-x:auto"><table class="data-table"><thead><tr><th>Engine</th><th>Trades</th><th>WR</th><th>Avg Ret</th><th>Sharpe</th><th>PF</th></tr></thead><tbody>';
        engineNames.forEach(function (name) {
          var es = perEngine[name].summary || {};
          var pnlClass = (es.avg_return_pct || 0) > 0 ? 'positive' : ((es.avg_return_pct || 0) < 0 ? 'negative' : '');
          html += '<tr><td><b>' + escapeHtml(engineDisplayName(name)) + '</b></td><td>' + (es.total_trades || 0) + '</td>' +
            '<td>' + fmtPct((es.win_rate || 0) * 100) + '</td><td class="' + pnlClass + '">' + fmtPct(es.avg_return_pct) + '</td>' +
            '<td>' + fmt(es.sharpe_ratio) + '</td><td>' + fmt(es.profit_factor) + '</td></tr>';
        });
        html += '</tbody></table></div></div>';
      }

      detail.innerHTML = html;
    }).catch(function () { detail.innerHTML = '<div class="empty-state"><p>Failed to load backtest.</p></div>'; });
  }

  // =======================================================================
  // SYSTEM TAB (merged: Pipeline + Costs + Histories)
  // =======================================================================
  function loadSystem() {
    showSpinner('system-view');

    Promise.all([
      fetchJSON('/api/runs?limit=30'),
      fetchJSON('/api/costs?days=30').catch(function () { return { total_cost_usd: 0, by_agent: {}, by_date: {} }; }),
      fetchJSON('/api/cache-stats').catch(function () { return { hit_rate: 0, hits: 0, misses: 0, total_entries: 0 }; }),
    ]).then(function (results) {
      var runs = results[0] || [];
      var costs = results[1];
      var cache = results[2];
      var view = document.getElementById('system-view');

      var html = '<div class="subtab-bar">' +
        '<button class="subtab-btn subtab-active" data-subtab="sys-pipeline">Pipeline</button>' +
        '<button class="subtab-btn" data-subtab="sys-history">History</button>' +
        '<button class="subtab-btn" data-subtab="sys-costs">Costs</button>' +
        '</div>';

      // --- Pipeline ---
      html += '<div class="subtab-content subtab-visible" data-subtab-content="sys-pipeline">';
      if (runs.length === 0) {
        html += '<div class="empty-state"><p>No pipeline runs recorded yet.</p></div>';
      } else {
        var pRows = runs.map(function (r) {
          return '<tr><td>' + escapeHtml(r.run_date) + '</td><td>' + regimeBadge(r.regime) + '</td>' +
            '<td>' + (r.universe_size || '\u2014') + '</td><td>' + (r.candidates_scored || '\u2014') + '</td>' +
            '<td>' + (r.pipeline_duration_s != null ? fmt(r.pipeline_duration_s, 1) + 's' : '\u2014') + '</td></tr>';
        }).join('');
        html += '<div class="card"><div class="card-title" style="margin-bottom:0.75rem">Recent Runs</div>' +
          '<table class="data-table"><thead><tr><th>Date</th><th>Regime</th><th>Universe</th><th>Scored</th><th>Duration</th></tr></thead>' +
          '<tbody>' + pRows + '</tbody></table></div>';
      }
      html += '</div>';

      // --- History ---
      html += '<div class="subtab-content" data-subtab-content="sys-history">';
      if (runs.length === 0) {
        html += '<div class="empty-state"><p>No historical runs.</p></div>';
      } else {
        var hRows = runs.map(function (r) {
          var reportUrl = '/report/' + encodeURIComponent(r.run_date);
          return '<tr><td><a class="report-link" href="' + reportUrl + '">' + escapeHtml(r.run_date) + '</a></td>' +
            '<td>' + regimeBadge(r.regime) + '</td><td>' + (r.candidates_scored || '\u2014') + '</td>' +
            '<td>' + (r.pipeline_duration_s != null ? fmt(r.pipeline_duration_s, 1) + 's' : '\u2014') + '</td></tr>';
        }).join('');
        html += '<div class="card"><div class="card-title" style="margin-bottom:0.75rem">Run History</div>' +
          '<p class="card-subtitle" style="margin-bottom:1rem">Click any date to view its full report.</p>' +
          '<table class="data-table"><thead><tr><th>Date</th><th>Regime</th><th>Scored</th><th>Duration</th></tr></thead>' +
          '<tbody>' + hRows + '</tbody></table></div>';
      }
      html += '</div>';

      // --- Costs ---
      html += '<div class="subtab-content" data-subtab-content="sys-costs">';
      html += '<div class="metrics-grid">' +
        metricCard('$' + fmt(costs.total_cost_usd, 4), 'Total Cost (30d)', null) +
        metricCard(numberFormat(costs.total_tokens_in), 'Tokens In', null) +
        metricCard(numberFormat(costs.total_tokens_out), 'Tokens Out', null) +
        metricCard(cache.hits === 0 && cache.misses === 0 ? 'N/A' : fmtPct(cache.hit_rate * 100), 'Cache Hit Rate', cache.hit_rate > 0.5) +
        '</div>';

      var nonZeroAgents = costs.by_agent ? Object.keys(costs.by_agent).filter(function (a) { return costs.by_agent[a] > 0; }) : [];
      if (nonZeroAgents.length > 0) {
        var agentRows = nonZeroAgents.map(function (a) {
          return '<tr><td>' + escapeHtml(a) + '</td><td>$' + fmt(costs.by_agent[a], 4) + '</td></tr>';
        }).join('');
        html += '<div class="card"><div class="card-title" style="margin-bottom:0.75rem">Cost by Agent</div>' +
          '<table class="data-table"><thead><tr><th>Agent</th><th>Cost</th></tr></thead><tbody>' + agentRows + '</tbody></table></div>';
      }

      if (costs.by_date && Object.keys(costs.by_date).length > 0) {
        var dateRows = Object.keys(costs.by_date).sort().reverse().map(function (d) {
          return '<tr><td>' + escapeHtml(d) + '</td><td>$' + fmt(costs.by_date[d], 4) + '</td></tr>';
        }).join('');
        html += '<div class="card"><div class="card-title" style="margin-bottom:0.75rem">Cost by Date</div>' +
          '<table class="data-table"><thead><tr><th>Date</th><th>Cost</th></tr></thead><tbody>' + dateRows + '</tbody></table></div>';
      }
      html += '</div>';

      view.innerHTML = html;
      initSubTabs('system-view');
    }).catch(function () {
      showEmpty('system-view', 'Failed to load system data.');
    });
  }

  // -----------------------------------------------------------------------
  // Boot: load default tab
  // -----------------------------------------------------------------------
  switchTab('overview');

})();
