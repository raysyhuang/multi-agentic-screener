/* Dashboard SPA — vanilla JS with tab routing and lazy data loading */

(function () {
  'use strict';

  // -----------------------------------------------------------------------
  // Engine display names — avoids naive title-casing of acronyms
  // -----------------------------------------------------------------------
  var ENGINE_DISPLAY = { gemini_stst: 'Gemini STST', koocore_d: 'KooCore-D' };
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
    signals: false, crossengine: false, performance: false,
    charts: false, compare: false, pipeline: false, costs: false, histories: false,
    backtest: false
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
      case 'histories': loadHistories(); break;
      case 'backtest': loadBacktest(); break;
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
    html += '<div style="font-size:0.75rem;color:var(--text-muted);margin-bottom:0.35rem">' +
      'Showing issues first. Passing checks are collapsed.' +
      '</div>';
    if (problemChecks.length === 0) {
      html += '<div style="font-size:0.8rem;color:var(--text-secondary)">✅ All ' + passChecks.length + ' dataset checks passed.</div>';
    } else {
      problemChecks.forEach(function (c) {
        var icon = c.passed ? '\u2705' : '\u26A0\uFE0F';
        html += '<div style="font-size:0.8rem;margin:0.25rem 0;color:var(--text-secondary)">' +
          icon + ' <strong>' + escapeHtml(c.name.replace(/_/g, ' ')) + '</strong>: ' +
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

    var totalStageChecks = 0;
    var totalStageProblems = 0;
    stages.forEach(function (stage) { totalStageChecks += (stage.checks || []).length; });
    stages.forEach(function (stage) {
      (stage.checks || []).forEach(function (c) { if (!c.passed) totalStageProblems++; });
    });
    html += '<div style="font-size:0.75rem;color:var(--text-muted);margin-bottom:0.35rem">' +
      'Showing stage summaries and only checks that need attention.' +
      '</div>';
    if (totalStageProblems === 0) {
      html += '<div style="font-size:0.8rem;color:var(--text-secondary);margin-bottom:0.35rem">✅ All ' + totalStageChecks + ' pipeline checks passed.</div>';
    }

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

    var statusMap = {};
    if (endpointCheck.value && endpointCheck.value.endpoints) {
      statusMap = endpointCheck.value.endpoints;
    }
    if (!statusMap || Object.keys(statusMap).length === 0) return '';

    function badge(status) {
      if (status === 'supported') return { color: 'var(--green)', label: 'supported' };
      if (status === 'plan_gated') return { color: 'var(--amber, #f59e0b)', label: 'plan-gated' };
      if (status === 'unsupported') return { color: 'var(--amber, #f59e0b)', label: 'unsupported' };
      if (status === 'disabled') return { color: 'var(--red, #ef4444)', label: 'disabled' };
      if (status === 'auth_error') return { color: 'var(--red, #ef4444)', label: 'auth-error' };
      return { color: 'var(--text-muted)', label: status || 'unknown' };
    }

    var html = '<div class="card" style="margin-bottom:1.25rem">' +
      '<div class="card-header">' +
      '<div><span class="card-title">FMP Endpoint Availability</span>' +
      '<div class="card-subtitle">Supported vs plan-gated endpoints for this run</div></div>' +
      '</div>';

    var callsUsed = endpointCheck.value.calls_used;
    var dailyBudget = endpointCheck.value.daily_budget;
    if (callsUsed != null && dailyBudget != null) {
      html += '<div style="font-size:0.74rem;color:var(--text-muted);margin-bottom:0.45rem">Calls used: ' +
        escapeHtml(String(callsUsed)) + '/' + escapeHtml(String(dailyBudget)) + '</div>';
    }

    html += '<div style="display:flex;flex-wrap:wrap;gap:0.4rem">';
    Object.keys(statusMap).forEach(function (name) {
      var meta = badge(statusMap[name]);
      html += '<span style="font-size:0.72rem;border:1px solid ' + meta.color + ';color:' + meta.color +
        ';border-radius:999px;padding:0.12rem 0.42rem">' +
        escapeHtml(name.replace(/_/g, ' ')) + ': ' + escapeHtml(meta.label) + '</span>';
    });
    html += '</div>';

    if (endpointCheck.message) {
      html += '<div style="font-size:0.74rem;color:var(--text-muted);margin-top:0.45rem">' +
        escapeHtml(endpointCheck.message) + '</div>';
    }
    html += '</div>';
    return html;
  }

  function renderVerificationChecklist(pipelineHealth, datasetHealth) {
    var stageItems = [];
    var checkItems = [];
    var datasetItems = [];
    var warningItems = [];

    function normalizeStatus(passed, severity) {
      var sev = (severity || '').toLowerCase();
      if (passed) return 'pass';
      if (sev === 'fail') return 'fail';
      return 'warn';
    }

    function statusView(status) {
      if (status === 'pass') return { icon: '\u2705', color: 'var(--green)', label: 'PASS' };
      if (status === 'fail') return { icon: '\u274C', color: 'var(--red, #ef4444)', label: 'FAIL' };
      return { icon: '\u26A0\uFE0F', color: 'var(--amber, #f59e0b)', label: 'WARN' };
    }

    function renderRows(items) {
      if (!items.length) {
        return '<div style="font-size:0.78rem;color:var(--text-muted);padding:0.3rem 0">No checks in this section.</div>';
      }
      var section = '';
      items.forEach(function (item) {
        var view = statusView(item.status);
        section += '<div style="padding:0.42rem 0.52rem;border:1px solid var(--border);border-radius:8px;margin-top:0.35rem">' +
          '<div style="display:flex;align-items:center;justify-content:space-between;gap:0.5rem">' +
          '<div style="display:flex;align-items:center;gap:0.35rem;font-size:0.78rem;color:var(--text-secondary)">' +
          '<span style="color:' + view.color + '">' + view.icon + '</span>' +
          '<strong>' + escapeHtml(item.label) + '</strong>' +
          '</div>' +
          '<span style="font-size:0.68rem;color:' + view.color + ';border:1px solid ' + view.color + ';border-radius:999px;padding:0.05rem 0.35rem">' + view.label + '</span>' +
          '</div>' +
          '<div style="font-size:0.74rem;color:var(--text-muted);margin-top:0.18rem">' + escapeHtml(item.detail || '') + '</div>' +
          '</div>';
      });
      return section;
    }

    function renderSection(title, items, opts) {
      opts = opts || {};
      var problemItems = items.filter(function (i) { return i.status !== 'pass'; });
      var passItems = items.filter(function (i) { return i.status === 'pass'; });
      var open = problemItems.length > 0 || !!opts.forceOpen;
      var summarySuffix = '';
      if (problemItems.length === 0 && items.length > 0) {
        summarySuffix = ' \u00B7 all pass';
      } else if (problemItems.length > 0) {
        summarySuffix = ' \u00B7 ' + problemItems.length + ' issues';
      }
      var html = '<details' + (open ? ' open' : '') + (opts.marginTop ? ' style="margin-top:' + opts.marginTop + '"' : '') + '>' +
        '<summary style="cursor:pointer;font-size:0.8rem;color:var(--text-secondary);font-weight:600">' +
        title + ' (' + items.length + ')' + summarySuffix +
        '</summary>';

      if (items.length === 0) {
        html += '<div style="font-size:0.78rem;color:var(--text-muted);padding:0.3rem 0">No checks in this section.</div>';
      } else if (problemItems.length === 0) {
        html += '<div style="font-size:0.78rem;color:var(--text-muted);padding:0.35rem 0">All ' + passItems.length + ' checks passed.</div>';
      } else {
        html += renderRows(problemItems);
      }

      if (passItems.length > 0) {
        html += '<details style="margin-top:0.35rem"><summary style="cursor:pointer;font-size:0.75rem;color:var(--text-secondary)">Show passing rows (' + passItems.length + ')</summary>' +
          renderRows(passItems) +
          '</details>';
      }

      html += '</details>';
      return html;
    }

    if (pipelineHealth && pipelineHealth.stages) {
      (pipelineHealth.stages || []).forEach(function (stage) {
        var stageLabel = (stage.stage || '').replace(/_/g, ' ');
        var stageStatus = normalizeStatus(!!stage.passed, stage.severity || 'warn');
        stageItems.push({
          label: stageLabel,
          status: stageStatus,
          detail: stage.passed ? 'stage passed' : 'stage requires review',
        });

        (stage.checks || []).forEach(function (c) {
          checkItems.push({
            label: stageLabel + ' / ' + (c.name || '').replace(/_/g, ' '),
            status: normalizeStatus(!!c.passed, c.severity || 'warn'),
            detail: c.message || '',
          });
        });
      });
    }

    if (datasetHealth && datasetHealth.checks) {
      (datasetHealth.checks || []).forEach(function (c) {
        datasetItems.push({
          label: (c.name || '').replace(/_/g, ' '),
          status: c.passed ? 'pass' : 'warn',
          detail: c.detail || '',
        });
      });
    }

    if (pipelineHealth && pipelineHealth.warnings) {
      (pipelineHealth.warnings || []).forEach(function (w) {
        warningItems.push({ label: 'Pipeline warning', status: 'warn', detail: w || '' });
      });
    }

    var allItems = stageItems.concat(checkItems).concat(datasetItems).concat(warningItems);
    var passCount = allItems.filter(function (i) { return i.status === 'pass'; }).length;
    var warnCount = allItems.filter(function (i) { return i.status === 'warn'; }).length;
    var failCount = allItems.filter(function (i) { return i.status === 'fail'; }).length;

    var html = '<div class="card" style="margin-bottom:1.25rem">' +
      '<div class="card-header">' +
      '<div><span class="card-title">Verification Checklist</span>' +
      '<div class="card-subtitle">Complete run checklist for morning verification</div></div>' +
      '</div>' +
      '<div style="display:flex;flex-wrap:wrap;gap:0.45rem;margin-bottom:0.6rem">' +
      '<span style="font-size:0.72rem;color:var(--text-secondary);border:1px solid var(--border);border-radius:999px;padding:0.12rem 0.45rem">Total: ' + allItems.length + '</span>' +
      '<span style="font-size:0.72rem;color:var(--green);border:1px solid var(--green);border-radius:999px;padding:0.12rem 0.45rem">PASS: ' + passCount + '</span>' +
      '<span style="font-size:0.72rem;color:var(--amber, #f59e0b);border:1px solid var(--amber, #f59e0b);border-radius:999px;padding:0.12rem 0.45rem">WARN: ' + warnCount + '</span>' +
      '<span style="font-size:0.72rem;color:var(--red, #ef4444);border:1px solid var(--red, #ef4444);border-radius:999px;padding:0.12rem 0.45rem">FAIL: ' + failCount + '</span>' +
      '</div>' +
      '<div style="font-size:0.74rem;color:var(--text-muted);margin-bottom:0.6rem">' +
      'Row status reflects pass/fail outcome. Check severity indicates how serious the issue would be if that check failed.' +
      '</div>' +
      renderSection('Pipeline Stages', stageItems, { forceOpen: true }) +
      renderSection('Pipeline Checks', checkItems, { marginTop: '0.45rem' }) +
      renderSection('Dataset Checks', datasetItems, { marginTop: '0.45rem' }) +
      renderSection('Warnings', warningItems, { marginTop: '0.45rem' }) +
      '</div>';

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
        var dropStats = ceh.engine_failure_stats || null;
        var dropLine = '';
        if (dropStats) {
          var totalDropped = Number(dropStats.total_failed || 0);
          var byKind = dropStats.by_kind || {};
          var kindParts = [];
          Object.keys(byKind).forEach(function (kind) {
            var count = Number(byKind[kind] || 0);
            if (count > 0) kindParts.push(escapeHtml(kind) + ': ' + count);
          });
          dropLine =
            '<div style="margin-top:0.6rem;font-size:0.8rem;color:var(--text-secondary)">' +
            '<strong>Engine Drop Metrics:</strong> ' + totalDropped +
            (kindParts.length ? ' (' + kindParts.join(', ') + ')' : '') +
            '</div>';
        }
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
        html += '</div></div>' + dropLine + '</div>';
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
          var displayName = engineDisplayName(name);

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

          // Strategy tags & signal count
          var stratTags = pos.strategy_tags || pos.strategies || [];
          var effSignals = pos.effective_signal_count || 0;
          var regimeWt = pos.regime_weight || null;

          var stratHtml = '';
          if (Array.isArray(stratTags) && stratTags.length > 0) {
            stratHtml = '<div class="strategy-tags-row">';
            stratTags.forEach(function (tag) {
              var cls = tag.indexOf('kc_') === 0 ? 'tag-kc' : (tag.indexOf('gem_') === 0 ? 'tag-gem' : 'tag-other');
              stratHtml += '<span class="strategy-tag ' + cls + '">' + escapeHtml(tag) + '</span>';
            });
            stratHtml += '</div>';
          }

          var signalInfo = '';
          if (effSignals > 0) {
            signalInfo = fmt(effSignals, 1) + ' signals';
          }
          var regimeInfo = regimeWt ? ' \u2022 regime ' + fmt(regimeWt, 2) + 'x' : '';

          html += '<div class="portfolio-card">' +
            '<div class="portfolio-card-header">' +
              '<span class="portfolio-ticker">' + escapeHtml(ticker) + adjTag + '</span>' +
              '<span class="portfolio-weight">' + fmt(weight, 0) + '%</span>' +
            '</div>' +
            stratHtml +
            '<div class="portfolio-detail">' +
              '<div class="portfolio-detail-item"><div class="portfolio-detail-label">Entry</div><div class="portfolio-detail-value">$' + fmt(entry) + '</div></div>' +
              '<div class="portfolio-detail-item"><div class="portfolio-detail-label">Target</div><div class="portfolio-detail-value positive">+' + fmt(rewardPct, 1) + '%</div></div>' +
              '<div class="portfolio-detail-item"><div class="portfolio-detail-label">Risk</div><div class="portfolio-detail-value negative">' + fmt(riskPct, 1) + '%</div></div>' +
            '</div>' +
            '<div style="margin-top:0.5rem;font-size:0.7rem;color:var(--text-muted)">' +
              (signalInfo ? signalInfo + ' \u2022 ' : '') +
              'Stop $' + fmt(stop) + ' \u2022 ' + hold + 'd hold' + regimeInfo +
              (source ? ' \u2022 ' + escapeHtml(source) : '') +
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
        '<p class="card-subtitle" style="margin-bottom:1rem">Compare Quant Only vs Hybrid vs Agentic Full performance</p>';

      var modeColors = { agentic_full: 'var(--teal-500)', hybrid: '#a855f7', quant_only: 'var(--green)' };
      var modeDisplay = { agentic_full: 'Agentic Full', hybrid: 'Hybrid', quant_only: 'Quant Only' };
      function modeName(m) { return modeDisplay[m] || m.replace(/_/g, ' '); }

      html += '<div class="metrics-grid">';
      data.comparison.forEach(function (m) {
        var pnlClass = m.avg_pnl > 0 ? 'positive' : (m.avg_pnl < 0 ? 'negative' : '');
        var borderColor = modeColors[m.mode] || 'var(--teal-500)';
        html += '<div class="metric-card" style="border-top:3px solid ' + borderColor + '">' +
          '<div class="metric-value">' + escapeHtml(modeName(m.mode || 'unknown')) + '</div>' +
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
          '<td><b>' + escapeHtml(modeName(m.mode || '?')) + '</b></td>' +
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
  // Histories Tab
  // -----------------------------------------------------------------------
  function loadHistories() {
    showSpinner('histories-view');
    fetchJSON('/api/runs?limit=100').then(function (runs) {
      var view = document.getElementById('histories-view');

      if (!runs || runs.length === 0) {
        showEmpty('histories-view', 'No historical runs recorded yet.');
        return;
      }

      var rows = runs.map(function (r) {
        var reportUrl = '/report/' + encodeURIComponent(r.run_date);
        return '<tr>' +
          '<td><a class="report-link" href="' + reportUrl + '">' + escapeHtml(r.run_date) + '</a></td>' +
          '<td>' + regimeBadge(r.regime) + '</td>' +
          '<td>' + (r.candidates_scored || '\u2014') + '</td>' +
          '<td>' + (r.pipeline_duration_s != null ? fmt(r.pipeline_duration_s, 1) + 's' : '\u2014') + '</td>' +
        '</tr>';
      }).join('');

      view.innerHTML = '<div class="card">' +
        '<div class="card-title" style="margin-bottom:0.5rem">Run History</div>' +
        '<p class="card-subtitle" style="margin-bottom:1rem">Open any date to view its full daily report.</p>' +
        '<table class="data-table">' +
          '<thead><tr><th>Date</th><th>Regime</th><th>Scored</th><th>Duration</th></tr></thead>' +
          '<tbody>' + rows + '</tbody>' +
        '</table>' +
      '</div>';
    }).catch(function () {
      showEmpty('histories-view', 'Failed to load historical runs.');
    });
  }

  // -----------------------------------------------------------------------
  // Costs Tab
  // -----------------------------------------------------------------------
  function loadCosts() {
    showSpinner('costs-view');

    Promise.all([
      fetchJSON('/api/costs?days=30'),
      fetchJSON('/api/cache-stats').catch(function () {
        return { hit_rate: 0, hits: 0, misses: 0, total_entries: 0, evictions: 0 };
      }),
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
  // Backtest Tab
  // -----------------------------------------------------------------------
  var backtestChart = null;
  var backtestRuns = [];
  var backtestCompareMode = false;

  function loadBacktest() {
    showSpinner('backtest-view');
    fetchJSON('/api/dashboard/backtest/runs').then(function (data) {
      var view = document.getElementById('backtest-view');
      backtestRuns = data.runs || [];

      if (backtestRuns.length === 0) {
        showEmpty('backtest-view', 'No backtest results found. Run a multi-engine backtest first.');
        return;
      }

      var options = backtestRuns.map(function (r) {
        var range = r.date_range ? (r.date_range.start + ' to ' + r.date_range.end) : '';
        var label = range + ' | ' + (r.trading_days || 0) + ' days | ' +
          (r.total_trades_all_tracks || 0) + ' trades';
        return '<option value="' + escapeHtml(r.filename) + '">' + escapeHtml(label) + '</option>';
      }).join('');

      var html = '<select class="backtest-select" id="backtest-run-select">' +
        '<option value="">Select a backtest run...</option>' + options + '</select>' +
        '<div id="backtest-compare-controls" style="display:none;margin-bottom:1rem">' +
          '<select class="backtest-select" id="backtest-compare-select">' +
            '<option value="">Select run to compare...</option>' + options + '</select>' +
        '</div>' +
        '<div style="margin-bottom:1rem">' +
          '<button id="backtest-compare-toggle" style="background:none;border:1px solid var(--card-border);' +
          'border-radius:8px;padding:0.4rem 0.8rem;cursor:pointer;color:var(--text-secondary);' +
          'font-size:0.8rem;font-family:inherit">Compare Runs</button>' +
        '</div>' +
        '<div id="backtest-detail"></div>';

      view.innerHTML = html;

      document.getElementById('backtest-run-select').addEventListener('change', function () {
        var fname = this.value;
        if (!fname) { document.getElementById('backtest-detail').innerHTML = ''; return; }
        loadBacktestDetail(fname);
      });

      // Attach compare-select listener once (outside toggle) to avoid accumulation
      document.getElementById('backtest-compare-select').addEventListener('change', function () {
        var mainFile = document.getElementById('backtest-run-select').value;
        var compareFile = this.value;
        if (mainFile && compareFile && mainFile !== compareFile) {
          loadBacktestCompare(mainFile, compareFile);
        }
      });

      document.getElementById('backtest-compare-toggle').addEventListener('click', function () {
        backtestCompareMode = !backtestCompareMode;
        var ctrl = document.getElementById('backtest-compare-controls');
        ctrl.style.display = backtestCompareMode ? 'block' : 'none';
        this.textContent = backtestCompareMode ? 'Hide Compare' : 'Compare Runs';
      });
    }).catch(function () {
      showEmpty('backtest-view', 'Failed to load backtest runs.');
    });
  }

  function loadBacktestDetail(filename) {
    var detail = document.getElementById('backtest-detail');
    detail.innerHTML = '<div class="spinner">Loading...</div>';

    fetchJSON('/api/dashboard/backtest/' + encodeURIComponent(filename)).then(function (data) {
      var html = '';

      // Header card
      var range = data.date_range ? (data.date_range.start + ' to ' + data.date_range.end) : '';
      var elapsed = data.elapsed_s ? fmt(data.elapsed_s, 1) + 's' : '\u2014';
      var engines = (data.engines || []).join(', ');
      html += '<div class="card" style="margin-bottom:1.25rem">' +
        '<div class="card-header"><div>' +
          '<span class="card-title">Backtest Results</span>' +
          '<div class="card-subtitle">' + escapeHtml(range) + '</div>' +
        '</div></div>' +
        '<div class="metrics-grid" style="margin-bottom:0">' +
          metricCard(escapeHtml(data.run_date || ''), 'Run Date', null) +
          metricCard(data.trading_days || 0, 'Trading Days', null) +
          metricCard(elapsed, 'Elapsed', null) +
          metricCard(escapeHtml(engines), 'Engines', null) +
        '</div></div>';

      // Synthesis metrics — prefer credibility_weight, fallback to equal_weight
      var synthTrack = (data.synthesis || {}).credibility_weight || (data.synthesis || {}).equal_weight;
      var synthName = (data.synthesis || {}).credibility_weight ? 'credibility_weight' : 'equal_weight';
      if (synthTrack && synthTrack.summary) {
        var s = synthTrack.summary;
        html += '<div class="section-header">' +
          '<div class="section-icon">\uD83D\uDCC8</div>' +
          '<span class="section-title">Synthesis Metrics (' + synthName.replace(/_/g, ' ') + ')</span></div>';
        html += '<div class="card" style="margin-bottom:1rem;background:rgba(245,158,11,0.06);border-left:3px solid rgba(245,158,11,0.5)">' +
          '<div style="font-size:0.82rem;color:var(--text-secondary)">Backtest totals/drawdowns are shown as cumulative trade PnL points (sum of trade % returns), not capital-equity % returns.</div>' +
          '</div>';
        html += '<div class="metrics-grid">' +
          metricCard(s.total_trades || 0, 'Total Trades', null) +
          metricCard(fmtPct((s.win_rate || 0) * 100), 'Win Rate', s.win_rate >= 0.5) +
          metricCard(fmtPct(s.avg_return_pct), 'Avg Return', s.avg_return_pct > 0) +
          metricCard(fmt(s.sharpe_ratio), 'Sharpe', s.sharpe_ratio > 0) +
          metricCard(fmt(s.sortino_ratio), 'Sortino', s.sortino_ratio > 0) +
          metricCard(fmtPct(s.max_drawdown_pct), 'Trade DD (Pts)', false) +
          metricCard(fmt(s.profit_factor), 'Profit Factor', s.profit_factor > 1) +
          metricCard(fmt(s.expectancy), 'Expectancy', s.expectancy > 0) +
        '</div>';
      }

      // Per-engine summary table
      var perEngine = data.per_engine || {};
      var engineNames = Object.keys(perEngine);
      if (engineNames.length > 0) {
        html += '<div class="card">' +
          '<div class="card-title" style="margin-bottom:0.75rem">Per-Engine Summary</div>' +
          '<div style="overflow-x:auto"><table class="data-table">' +
          '<thead><tr><th>Engine</th><th>Trades</th><th>Win Rate</th><th>Avg Return</th>' +
          '<th>Sharpe</th><th>Profit Factor</th><th>Expectancy</th><th>Trade DD (Pts)</th></tr></thead><tbody>';
        engineNames.forEach(function (name) {
          var es = perEngine[name].summary || {};
          var pnlClass = (es.avg_return_pct || 0) > 0 ? 'positive' : ((es.avg_return_pct || 0) < 0 ? 'negative' : '');
          html += '<tr>' +
            '<td><b>' + escapeHtml(engineDisplayName(name)) + '</b></td>' +
            '<td>' + (es.total_trades || 0) + '</td>' +
            '<td>' + fmtPct((es.win_rate || 0) * 100) + '</td>' +
            '<td class="' + pnlClass + '">' + fmtPct(es.avg_return_pct) + '</td>' +
            '<td>' + fmt(es.sharpe_ratio) + '</td>' +
            '<td>' + fmt(es.profit_factor) + '</td>' +
            '<td>' + fmt(es.expectancy) + '</td>' +
            '<td class="negative">' + fmtPct(es.max_drawdown_pct) + '</td>' +
          '</tr>';
        });
        html += '</tbody></table></div></div>';
      }

      // Synthesis tracks table
      var synthTracks = data.synthesis || {};
      var trackNames = Object.keys(synthTracks);
      if (trackNames.length > 0) {
        html += '<div class="card">' +
          '<div class="card-title" style="margin-bottom:0.75rem">Synthesis Tracks</div>' +
          '<div style="overflow-x:auto"><table class="data-table">' +
          '<thead><tr><th>Track</th><th>Trades</th><th>Win Rate</th><th>Avg Return</th>' +
          '<th>Sharpe</th><th>Profit Factor</th><th>Expectancy</th><th>Trade DD (Pts)</th></tr></thead><tbody>';
        trackNames.forEach(function (name) {
          var ts = synthTracks[name].summary || {};
          var pnlClass = (ts.avg_return_pct || 0) > 0 ? 'positive' : ((ts.avg_return_pct || 0) < 0 ? 'negative' : '');
          html += '<tr>' +
            '<td><b>' + escapeHtml(name.replace(/_/g, ' ')) + '</b></td>' +
            '<td>' + (ts.total_trades || 0) + '</td>' +
            '<td>' + fmtPct((ts.win_rate || 0) * 100) + '</td>' +
            '<td class="' + pnlClass + '">' + fmtPct(ts.avg_return_pct) + '</td>' +
            '<td>' + fmt(ts.sharpe_ratio) + '</td>' +
            '<td>' + fmt(ts.profit_factor) + '</td>' +
            '<td>' + fmt(ts.expectancy) + '</td>' +
            '<td class="negative">' + fmtPct(ts.max_drawdown_pct) + '</td>' +
          '</tr>';
        });
        html += '</tbody></table></div></div>';
      }

      // By-regime breakdown (from synthesis credibility_weight)
      if (synthTrack && synthTrack.by_regime) {
        var regimes = synthTrack.by_regime;
        var regimeKeys = Object.keys(regimes);
        if (regimeKeys.length > 0) {
          html += '<div class="card">' +
            '<div class="card-title" style="margin-bottom:0.75rem">By Regime (' + synthName.replace(/_/g, ' ') + ')</div>' +
            '<table class="data-table">' +
            '<thead><tr><th>Regime</th><th>Trades</th><th>Win Rate</th><th>Avg Return</th></tr></thead><tbody>';
          regimeKeys.forEach(function (regime) {
            var rd = regimes[regime];
            var pnlClass = (rd.avg_return_pct || 0) > 0 ? 'positive' : ((rd.avg_return_pct || 0) < 0 ? 'negative' : '');
            html += '<tr>' +
              '<td>' + regimeBadge(regime) + '</td>' +
              '<td>' + (rd.trades || 0) + '</td>' +
              '<td>' + fmtPct((rd.win_rate || 0) * 100) + '</td>' +
              '<td class="' + pnlClass + '">' + fmtPct(rd.avg_return_pct) + '</td>' +
            '</tr>';
          });
          html += '</tbody></table></div>';
        }
      }

      // By-strategy breakdown (across engines)
      var stratRows = [];
      engineNames.forEach(function (engName) {
        var byStrat = (perEngine[engName] || {}).by_strategy || {};
        Object.keys(byStrat).forEach(function (strat) {
          var sd = byStrat[strat];
          stratRows.push({ engine: engName, strategy: strat, data: sd });
        });
      });
      if (stratRows.length > 0) {
        html += '<div class="card">' +
          '<div class="card-title" style="margin-bottom:0.75rem">By Strategy (per Engine)</div>' +
          '<table class="data-table">' +
          '<thead><tr><th>Engine</th><th>Strategy</th><th>Trades</th><th>Win Rate</th><th>Avg Return</th></tr></thead><tbody>';
        stratRows.forEach(function (row) {
          var sd = row.data;
          var pnlClass = (sd.avg_return_pct || 0) > 0 ? 'positive' : ((sd.avg_return_pct || 0) < 0 ? 'negative' : '');
          html += '<tr>' +
            '<td>' + escapeHtml(row.engine) + '</td>' +
            '<td>' + escapeHtml(row.strategy) + '</td>' +
            '<td>' + (sd.trades || 0) + '</td>' +
            '<td>' + fmtPct((sd.win_rate || 0) * 100) + '</td>' +
            '<td class="' + pnlClass + '">' + fmtPct(sd.avg_return_pct) + '</td>' +
          '</tr>';
        });
        html += '</tbody></table></div>';
      }

      // Equity curves button
      html += '<div style="margin-top:1rem">' +
        '<button id="backtest-equity-btn" style="background:var(--gradient-brand);color:#fff;border:none;' +
        'border-radius:10px;padding:0.6rem 1.2rem;cursor:pointer;font-size:0.85rem;font-family:inherit;font-weight:600">' +
        'Show Equity Curves</button></div>' +
        '<div id="backtest-equity-area" style="margin-top:1rem"></div>';

      detail.innerHTML = html;

      document.getElementById('backtest-equity-btn').addEventListener('click', function () {
        this.disabled = true;
        this.textContent = 'Loading...';
        loadBacktestEquity(filename);
      });
    }).catch(function () {
      detail.innerHTML = '<div class="empty-state"><p>Failed to load backtest detail.</p></div>';
    });
  }

  function loadBacktestEquity(filename) {
    var area = document.getElementById('backtest-equity-area');
    area.innerHTML = '<div class="spinner">Loading equity curves...</div>';

    fetchJSON('/api/dashboard/backtest/' + encodeURIComponent(filename) + '/equity').then(function (data) {
      var curves = data.curves || {};
      var curveNames = Object.keys(curves);
      if (curveNames.length === 0) {
        area.innerHTML = '<div class="empty-state"><p>No equity curve data.</p></div>';
        return;
      }

      var lineColors = ['#14b8a6', '#a855f7', '#f59e0b', '#22c55e', '#3b82f6', '#ef4444', '#ec4899', '#6366f1'];

      // Legend
      var legendHtml = '<div class="chart-legend">';
      curveNames.forEach(function (name, i) {
        var color = lineColors[i % lineColors.length];
        legendHtml += '<div class="chart-legend-item">' +
          '<span class="chart-legend-swatch" style="background:' + color + '"></span>' +
          '<span>' + escapeHtml(name.replace(/_/g, ' ')) + '</span></div>';
      });
      legendHtml += '</div>';

      area.innerHTML = legendHtml + '<div class="chart-container"><div id="backtest-equity-chart" style="height:400px"></div></div>';

      var el = document.getElementById('backtest-equity-chart');
      if (!el || typeof LightweightCharts === 'undefined') return;

      if (backtestChart) { backtestChart.remove(); backtestChart = null; }

      var c = chartColors();
      var chart = LightweightCharts.createChart(el, {
        width: el.clientWidth,
        height: 400,
        layout: { background: { color: c.bg }, textColor: c.text },
        grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } },
        rightPriceScale: { borderColor: c.border },
        timeScale: { borderColor: c.border },
      });
      backtestChart = chart;

      curveNames.forEach(function (name, i) {
        var color = lineColors[i % lineColors.length];
        var series = chart.addLineSeries({ color: color, lineWidth: 2, title: name });
        var pts = (curves[name] || []).map(function (p) {
          return { time: p.date, value: p.cumulative_pnl_pct };
        });
        series.setData(pts);
      });

      chart.timeScale().fitContent();
      window.addEventListener('resize', function () {
        if (backtestChart && el.clientWidth > 0) backtestChart.applyOptions({ width: el.clientWidth });
      });

      var btn = document.getElementById('backtest-equity-btn');
      if (btn) { btn.textContent = 'Show Equity Curves'; btn.disabled = false; }
    }).catch(function () {
      area.innerHTML = '<div class="empty-state"><p>Failed to load equity curves.</p></div>';
      var btn = document.getElementById('backtest-equity-btn');
      if (btn) { btn.textContent = 'Show Equity Curves'; btn.disabled = false; }
    });
  }

  function loadBacktestCompare(file1, file2) {
    var detail = document.getElementById('backtest-detail');
    detail.innerHTML = '<div class="spinner">Comparing runs...</div>';

    fetchJSON('/api/dashboard/backtest/compare?files=' + encodeURIComponent(file1 + ',' + file2)).then(function (data) {
      var cmp = data.comparison || [];
      if (cmp.length < 2) {
        detail.innerHTML = '<div class="empty-state"><p>Need 2 valid runs to compare.</p></div>';
        return;
      }

      var r1 = cmp[0], r2 = cmp[1];
      var html = '<div class="card"><div class="card-title" style="margin-bottom:0.75rem">Run Comparison</div>' +
        '<div style="font-size:0.8rem;color:var(--text-secondary);margin-bottom:0.75rem">Totals/drawdowns below are cumulative trade PnL points (not capital-equity %).</div>' +
        '<div style="overflow-x:auto"><table class="data-table"><thead><tr>' +
        '<th>Metric</th>' +
        '<th>' + escapeHtml((r1.date_range || {}).start || '') + ' to ' + escapeHtml((r1.date_range || {}).end || '') + '</th>' +
        '<th>' + escapeHtml((r2.date_range || {}).start || '') + ' to ' + escapeHtml((r2.date_range || {}).end || '') + '</th>' +
        '<th>Delta</th></tr></thead><tbody>';

      // Compare synthesis credibility_weight (or equal_weight)
      var s1 = (r1.synthesis || {}).credibility_weight || (r1.synthesis || {}).equal_weight || {};
      var s2 = (r2.synthesis || {}).credibility_weight || (r2.synthesis || {}).equal_weight || {};

      var compareMetrics = [
        { key: 'total_trades', label: 'Total Trades', fmt: function (v) { return v || 0; }, pct: false },
        { key: 'win_rate', label: 'Win Rate', fmt: function (v) { return fmtPct((v || 0) * 100); }, pct: true, mul: 100 },
        { key: 'avg_return_pct', label: 'Avg Return', fmt: function (v) { return fmtPct(v); }, pct: true },
        { key: 'sharpe_ratio', label: 'Sharpe', fmt: function (v) { return fmt(v); }, pct: false },
        { key: 'sortino_ratio', label: 'Sortino', fmt: function (v) { return fmt(v); }, pct: false },
        { key: 'max_drawdown_pct', label: 'Trade DD (Pts)', fmt: function (v) { return fmtPct(v); }, pct: true },
        { key: 'profit_factor', label: 'Profit Factor', fmt: function (v) { return fmt(v); }, pct: false },
        { key: 'expectancy', label: 'Expectancy', fmt: function (v) { return fmt(v); }, pct: false },
      ];

      compareMetrics.forEach(function (m) {
        var v1 = s1[m.key] || 0;
        var v2 = s2[m.key] || 0;
        var diff = v2 - v1;
        if (m.mul) { diff = diff * m.mul; }
        var deltaClass = diff > 0 ? 'positive' : (diff < 0 ? 'negative' : '');
        // For max drawdown, less negative is better so invert
        if (m.key === 'max_drawdown_pct') deltaClass = diff < 0 ? 'positive' : (diff > 0 ? 'negative' : '');
        var deltaStr = m.pct ? fmtPct(diff) : fmt(diff);
        html += '<tr><td><b>' + m.label + '</b></td>' +
          '<td>' + m.fmt(v1) + '</td>' +
          '<td>' + m.fmt(v2) + '</td>' +
          '<td class="' + deltaClass + '">' + deltaStr + '</td></tr>';
      });

      html += '</tbody></table></div></div>';

      // Per-engine comparison
      var allEngines = {};
      Object.keys(r1.per_engine || {}).forEach(function (e) { allEngines[e] = true; });
      Object.keys(r2.per_engine || {}).forEach(function (e) { allEngines[e] = true; });
      var engineList = Object.keys(allEngines);

      if (engineList.length > 0) {
        html += '<div class="card"><div class="card-title" style="margin-bottom:0.75rem">Per-Engine Comparison (Win Rate / Avg Return)</div>' +
          '<div style="overflow-x:auto"><table class="data-table"><thead><tr><th>Engine</th>' +
          '<th>Run 1 WR</th><th>Run 2 WR</th><th>WR Delta</th>' +
          '<th>Run 1 Avg</th><th>Run 2 Avg</th><th>Avg Delta</th></tr></thead><tbody>';
        engineList.forEach(function (eng) {
          var e1 = (r1.per_engine || {})[eng] || {};
          var e2 = (r2.per_engine || {})[eng] || {};
          var wr1 = (e1.win_rate || 0) * 100, wr2 = (e2.win_rate || 0) * 100;
          var wrDelta = wr2 - wr1;
          var wrDeltaClass = wrDelta > 0 ? 'positive' : (wrDelta < 0 ? 'negative' : '');
          var avg1 = e1.avg_return_pct || 0, avg2 = e2.avg_return_pct || 0;
          var avgDelta = avg2 - avg1;
          var avgDeltaClass = avgDelta > 0 ? 'positive' : (avgDelta < 0 ? 'negative' : '');
          html += '<tr><td><b>' + escapeHtml(eng) + '</b></td>' +
            '<td>' + fmtPct(wr1) + '</td><td>' + fmtPct(wr2) + '</td>' +
            '<td class="' + wrDeltaClass + '">' + fmtPct(wrDelta) + '</td>' +
            '<td>' + fmtPct(avg1) + '</td><td>' + fmtPct(avg2) + '</td>' +
            '<td class="' + avgDeltaClass + '">' + fmtPct(avgDelta) + '</td></tr>';
        });
        html += '</tbody></table></div></div>';
      }

      detail.innerHTML = html;
    }).catch(function () {
      detail.innerHTML = '<div class="empty-state"><p>Failed to load comparison.</p></div>';
    });
  }

  // -----------------------------------------------------------------------
  // Boot: load default tab
  // -----------------------------------------------------------------------
  switchTab('signals');

})();
