const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (character) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[character]));
const labels = {
  baseline: 'Frequency baseline', lightgbm: 'Boosted trees',
  regularized: 'Regularized presence', poisson: 'Poisson count',
  poisson_prize_90d: 'Poisson Prize 90d', ensemble: 'Ensemble',
};
const descriptions = {
  baseline: 'Ranks numbers by trailing frequency.',
  lightgbm: 'A boosted presence estimate for the next date.',
  regularized: 'A regularized presence model for comparison.',
  poisson: 'Repeat-inclusive expected occurrence count.',
  poisson_prize_90d: 'Prize-level LightGBM Poisson model trained on a rolling 90-day window.',
  ensemble: 'A blended view across available model scores.',
};
const chartColors = ['#315BE8', '#16A36A', '#E28A16', '#8B5CF6', '#D64A67'];
let sourceData = [];
let sourceRevision = null;
let runs = [];
let latestPayload = null;
let latestAnalytics = null;
let chartViewport = { start: 0, end: null };

for (const id of ['model', 'setupModel']) {
  const select = $(id);
  if (select && !select.querySelector('option[value="poisson_prize_90d"]')) {
    const option = document.createElement('option');
    option.value = 'poisson_prize_90d';
    option.textContent = 'Poisson Prize 90d';
    select.insertBefore(option, select.querySelector('option[value="all"]'));
  }
}

function setPage(view) {
  document.querySelectorAll('.page').forEach((item) => { item.hidden = item.id !== `page-${view}`; });
  document.querySelectorAll('[data-view]').forEach((button) => {
    button.classList.toggle('active', button.dataset.view === view);
  });
  const names = {
    overview: 'Overview', setup: 'Forecast setup', results: 'Results',
    compare: 'Compare models', analytics: 'Chart analysis', history: 'Run history',
    methodology: 'Methodology & safety',
  };
  $('crumbTitle').textContent = names[view] || 'Overview';
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function setConnection(text, offline = false) {
  $('connection').classList.toggle('offline', offline);
  $('connection').innerHTML = `<span class="status-dot"></span>${esc(text)}`;
}

function isoDay(date) { return date.toISOString().slice(0, 10); }
function addDays(isoDate, amount) {
  const date = new Date(`${isoDate}T00:00:00Z`);
  date.setUTCDate(date.getUTCDate() + amount);
  return isoDay(date);
}
function inclusiveDays(start, end) {
  return Math.floor((new Date(`${end}T00:00:00Z`) - new Date(`${start}T00:00:00Z`)) / 86400000) + 1;
}

function updateAnalyticsDates() {
  const source = sourceData[0];
  if (!source?.last_date) return;
  const period = $('analyticsPeriod').value;
  const length = period === 'custom' ? null : Number(period);
  for (const id of ['analyticsStart', 'analyticsEnd']) {
    $(id).min = source.first_date || '';
    $(id).max = source.last_date;
  }
  if (length) {
    $('analyticsEnd').value = source.last_date;
    $('analyticsStart').value = addDays(source.last_date, -(length - 1));
  }
}

function updateSource(item) {
  if (!item) return;
  const target = item.next_date || '';
  $('overviewSource').textContent = item.last_date ? `${item.label} · through ${item.last_date}` : item.label;
  $('sourceHint').textContent = item.last_date
    ? `${Number(item.days || 0).toLocaleString()} observed days · through ${item.last_date}`
    : 'Source metadata unavailable';
  $('target').value = target;
  $('target').max = target;
  $('targetEnd').value = '';
  $('targetEnd').max = target;
  $('targetEnd').min = target;
  $('latestMetric').textContent = item.last_date || '--';
  $('latestDetail').textContent = item.days ? `${Number(item.days).toLocaleString()} observed days` : 'Source unavailable';
  $('dataStatus').textContent = item.error || `${Number(item.days || 0).toLocaleString()} GitHub rows passed validation.`;
  $('readyStatus').textContent = target ? `Ready for target ${target}.` : 'Target date unavailable.';
  $('setupTarget').value = target;
  $('setupTarget').max = target;
  $('setupTargetEnd').value = '';
  $('setupTargetEnd').max = target;
  $('setupTargetEnd').min = target;
  $('setupDataCheck').textContent = item.error || `${Number(item.days || 0).toLocaleString()} GitHub rows passed validation.`;
  $('setupDateCheck').textContent = target ? `Next forecast date is ${target}.` : 'The GitHub feed is unavailable.';
  updateAnalyticsDates();
}

function fillSources() {
  const options = sourceData.map((source) => `<option value="${esc(source.id)}">${esc(source.label)}${source.last_date ? ` · through ${esc(source.last_date)}` : ''}</option>`).join('');
  $('source').innerHTML = options;
  $('setupSource').innerHTML = options;
  if (sourceData[0]) {
    $('source').value = sourceData[0].id;
    $('setupSource').value = sourceData[0].id;
    updateSource(sourceData[0]);
  }
}

async function loadSources() {
  const response = await fetch('/api/sources');
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || 'Unable to load the GitHub source.');
  sourceData = payload.sources || [];
  if (sourceData.length !== 1 || sourceData[0].id !== 'live') {
    throw new Error('The API must expose exactly the live GitHub data feed.');
  }
  fillSources();
  $('status').textContent = 'Ready to forecast.';
  setConnection('API ready');
}

async function refreshGithubSource() {
  const button = $('refreshSource');
  const originalText = button.textContent;
  $('status').classList.remove('error');
  button.disabled = true;
  button.textContent = 'Pulling latest data...';
  setConnection('Refreshing GitHub');
  try {
    const response = await fetch('/api/sources/refresh', { method: 'POST' });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'GitHub refresh failed.');
    sourceRevision = payload.source_revision;
    sourceData = [payload.source];
    fillSources();
    const refresh = payload.refresh || {};
    const message = refresh.changed
      ? `Added ${Number(refresh.added_days || 0)} day${Number(refresh.added_days || 0) === 1 ? '' : 's'}; latest draw is ${refresh.current_last_date}.`
      : `GitHub feed is already current through ${refresh.current_last_date}.`;
    $('status').classList.remove('error');
    $('status').textContent = message;
    $('dataStatus').textContent = message;
    $('analyticsStatus').textContent = message;
    setConnection('GitHub refreshed');
  } catch (error) {
    $('status').textContent = error.message;
    $('status').classList.add('error');
    setConnection('Refresh failed', true);
    throw error;
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
}

function isPrizeModel(id = 'model') { return $(id).value === 'poisson_prize_90d'; }
function syncRangeControls() {
  const mainPrize = isPrizeModel('model');
  const setupPrize = isPrizeModel('setupModel');
  $('targetEnd').disabled = !mainPrize;
  $('setupTargetEnd').disabled = !setupPrize;
  if (!mainPrize) $('targetEnd').value = '';
  if (!setupPrize) $('setupTargetEnd').value = '';
}
function getBody() {
  const prize = isPrizeModel();
  return {
    source: 'live', model: $('model').value, top_k: Number($('topk').value),
    test_days: Number($('days').value), target_date: $('target').value || null,
    target_end_date: prize ? ($('targetEnd').value || null) : null,
    source_revision: sourceRevision,
  };
}
async function postForecast(body) {
  const response = await fetch('/api/forecast', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || 'Forecast failed.');
  return payload;
}

function summarizeSeries(model) {
  const series = [...(model.series || [])].sort((left, right) => left.date.localeCompare(right.date));
  let cumulativeCost = 0;
  let cumulativeProfit = 0;
  let runningPeak = 0;
  let maxDrawdown = 0;
  let totalHits = 0;
  let daysWithHit = 0;
  let precision = 0;
  let recall = 0;
  let receipts = 0;
  series.forEach((row) => {
    cumulativeCost += Number(row.stake || 0);
    cumulativeProfit += Number(row.profit || 0);
    receipts += Number(row.receipts || 0);
    totalHits += Number(row.hits || 0);
    daysWithHit += Number(row.hits || 0) > 0 ? 1 : 0;
    precision += Number(row.precision_at_k || 0);
    recall += Number(row.recall_at_k || 0);
    runningPeak = Math.max(runningPeak, cumulativeProfit);
    maxDrawdown = Math.max(maxDrawdown, runningPeak - cumulativeProfit);
    row.cumulative_profit = cumulativeProfit;
    row.cumulative_roi = cumulativeCost ? cumulativeProfit / cumulativeCost : 0;
    row.drawdown = runningPeak - cumulativeProfit;
  });
  const days = series.length;
  return {
    ...model, series,
    summary: {
      days, days_with_hit: daysWithHit, hit_rate: days ? daysWithHit / days : 0,
      total_hits: totalHits, mean_hits: days ? totalHits / days : 0,
      precision_at_k: days ? precision / days : 0, recall_at_k: days ? recall / days : 0,
      total_stake: cumulativeCost, total_receipts: receipts, total_profit: cumulativeProfit,
      roi: cumulativeCost ? cumulativeProfit / cumulativeCost : 0, max_drawdown: maxDrawdown,
    },
  };
}

function mergeAnalytics(envelopes, selectedIds = null) {
  const models = new Map();
  envelopes.filter(Boolean).forEach((envelope) => {
    (envelope.models || []).forEach((model) => {
      if (selectedIds && !selectedIds.includes(model.model_id)) return;
      const existing = models.get(model.model_id) || {
        model_id: model.model_id,
        model_name: model.model_name || labels[model.model_id] || model.model_id,
        series: [],
      };
      existing.series.push(...(model.series || []));
      models.set(model.model_id, existing);
    });
  });
  const normalized = [...models.values()].map((model) => {
    const unique = new Map(model.series.map((row) => [row.date, row]));
    return summarizeSeries({ ...model, series: [...unique.values()] });
  });
  const dates = normalized.flatMap((model) => model.series.map((row) => row.date)).sort();
  return {
    schema_version: '1.0', granularity: 'day', currency: 'VND',
    period: { start_date: dates[0] || null, end_date: dates.at(-1) || null },
    payout_assumptions: {
      stake_per_selection: 27000, gross_payout_per_occurrence: 99000,
      payout_mode: 'per_occurrence',
    },
    models: normalized,
  };
}

async function fetchPrizeRange(body, progress) {
  const end = new Date(`${body.target_end_date}T00:00:00Z`);
  let cursor = new Date(`${body.target_date}T00:00:00Z`);
  const batches = [];
  const totalDays = Math.floor((end - cursor) / 86400000) + 1;
  const totalBatches = Math.ceil(totalDays / 31);
  let batchNumber = 0;
  while (cursor <= end) {
    batchNumber += 1;
    const batchEnd = new Date(cursor);
    batchEnd.setUTCDate(batchEnd.getUTCDate() + 30);
    if (batchEnd > end) batchEnd.setTime(end.getTime());
    progress?.(batchNumber, totalBatches);
    batches.push(await postForecast({
      ...body, target_date: isoDay(cursor), target_end_date: isoDay(batchEnd),
    }));
    cursor = new Date(batchEnd);
    cursor.setUTCDate(cursor.getUTCDate() + 1);
  }
  if (batches.length === 1) return batches[0];
  return {
    ...batches[0], target_start_date: body.target_date, target_end_date: body.target_end_date,
    result_count: batches.reduce((sum, batch) => sum + Number(batch.result_count || 0), 0),
    results: batches.flatMap((batch) => batch.results || []),
    analytics: mergeAnalytics(batches.map((batch) => batch.analytics)),
  };
}
async function fetchForecastPayload(body, progress) {
  return body.model === 'poisson_prize_90d' && body.target_date && body.target_end_date
    ? fetchPrizeRange(body, progress) : postForecast(body);
}

async function runForecast(body, statusId = 'status') {
  const status = $(statusId);
  status.textContent = 'Running chronological validation...';
  status.classList.remove('error');
  setConnection('Computing');
  try {
    const payload = await fetchForecastPayload(body, (batch, total) => {
      status.textContent = `Running batch ${batch} of ${total}...`;
    });
    latestPayload = payload;
    const items = Array.isArray(payload.results) ? payload.results : [payload];
    const target = payload.target_date || `${payload.target_start_date} to ${payload.target_end_date}`;
    const numbers = items.length === 1
      ? Object.values(items[0].forecasts || {}).flat().map((item) => item.number).join(', ')
      : `${items.length} daily forecasts`;
    runs.unshift({ date: new Date().toLocaleString(), model: payload.model, target, numbers, source: 'Live GitHub feed' });
    runs = runs.slice(0, 20);
    renderResults(payload);
    renderComparison(payload);
    renderHistory();
    setPage('results');
    status.textContent = `Forecast complete: ${items.length} date${items.length === 1 ? '' : 's'}.`;
    setConnection('API ready');
    return payload;
  } catch (error) {
    status.textContent = error.message || 'Request failed.';
    status.classList.add('error');
    setConnection('Needs attention', true);
    throw error;
  }
}

function renderResults(payload) {
  const isRange = Array.isArray(payload.results);
  const items = isRange ? payload.results : [payload];
  if (isRange) {
    const rows = items.map((item) => {
      const predictions = item.forecasts?.poisson_prize_90d || [];
      return `<tr><td><strong>${esc(item.target_date)}</strong></td><td>${predictions.map((entry) => `<span class="number">${esc(entry.number)}</span>`).join(' ')}</td><td>${predictions.map((entry) => Number(entry.score).toFixed(3)).join(' · ')}</td><td>${esc(item.data_through_date || '--')} (${esc(item.forecast_horizon_days || 1)}d)</td><td>${esc(item.training_start_date)} to ${esc(item.training_end_date)}</td></tr>`;
    }).join('');
    $('resultCards').innerHTML = `<article class="panel"><div class="panel-head"><div><h2>Daily walk-forward forecasts</h2><p>Each date is refitted independently using only its preceding 90-day label window.</p></div><span class="eyebrow">${items.length} dates</span></div><div class="table-wrap"><table><thead><tr><th>Target date</th><th>Ranked numbers</th><th>Scores</th><th>Data through</th><th>Training labels</th></tr></thead><tbody>${rows}</tbody></table></div></article>`;
  } else {
    const cards = Object.entries(payload.forecasts || {}).map(([model, predictions], index) => `<article class="result-card ${index === 0 ? 'featured' : ''}"><h3>${esc(labels[model] || model)}</h3><p>${esc(descriptions[model] || 'Primary forecast model.')}</p><div class="number-list">${predictions.map((entry) => `<span class="number">${esc(entry.number)}</span>`).join('')}</div><div class="score-row"><span>Rank scores</span><strong>${predictions.map((entry) => Number(entry.score).toFixed(3)).join(' · ')}</strong></div>${payload.evaluations?.[model]?.n_days ? `<div class="score-row" style="margin-top:10px"><span>Hit rate / ROI</span><strong>${(Number(payload.evaluations[model].hit_rate_at_k || 0) * 100).toFixed(1)}% / ${(Number(payload.payouts?.[model]?.roi || 0) * 100).toFixed(1)}%</strong></div>` : ''}</article>`).join('');
    $('resultCards').innerHTML = `<div class="result-grid">${cards || '<div class="empty"><strong>No forecast rows returned</strong></div>'}</div>`;
  }
  const targetText = isRange ? `${payload.target_start_date} to ${payload.target_end_date}` : (payload.target_date || '--');
  const cutoffText = isRange ? 'latest completed draw before each target' : (payload.data_through_date || payload.training_end_date || sourceData[0]?.last_date || 'source cutoff');
  $('resultMeta').textContent = `Target${isRange ? 's' : ''}: ${targetText} / Forecast cutoff: ${cutoffText} / ${payload.test_days || 30}-day window`;
  $('runId').textContent = `#XS-${new Date().toISOString().slice(2, 10).replaceAll('-', '')}-${String(runs.length).padStart(3, '0')}`;
  $('interpretation').innerHTML = `<div class="interpret-row"><span>Forecast cutoff</span><strong>${esc(cutoffText)}</strong></div><div class="interpret-row"><span>Training window</span><strong>${esc(payload.test_days || 30)} days</strong></div><div class="interpret-row"><span>Target dates</span><strong>${items.length}</strong></div><div class="interpret-row"><span>Ranking size</span><strong>${esc(payload.top_k || $('topk').value)}</strong></div>`;
  $('download').hidden = false;
  $('download').onclick = () => downloadCsv(payload);
}

function renderComparison(payload) {
  const rows = Object.entries(payload.evaluations || {});
  if (!rows.length) {
    $('compareAlert').textContent = 'Run “Compare all models” to populate a shared comparison.';
    $('agreementBars').innerHTML = '<div class="empty"><div class="empty-mark">≋</div><strong>Comparison is waiting</strong><p>Run all models to see agreement and holdout performance.</p></div>';
    $('dailyBars').innerHTML = '<div class="subtle">No evaluation run yet.</div>';
    $('comparisonTable').innerHTML = '<div class="empty"><strong>No comparison rows yet</strong><p>Use the all-models option on Forecast setup.</p></div>';
    return;
  }
  $('compareAlert').textContent = 'All models use the same GitHub source cutoff, dates, and ranking size.';
  $('agreementBars').innerHTML = rows.map(([name, item]) => `<div class="bar-row"><span>${esc(labels[name] || name)}</span><span class="bar-track"><span class="bar-fill" style="width:${Math.max(4, Math.min(100, Number(item.hit_rate_at_k || 0) * 100))}%"></span></span><strong class="bar-value">${(Number(item.hit_rate_at_k || 0) * 100).toFixed(0)}%</strong></div>`).join('');
  $('dailyBars').innerHTML = rows.map(([name, item]) => `<div class="perf-row"><strong>${esc(labels[name] || name)}</strong><span>${Number(item.mean_hits || 0).toFixed(2)} mean hits/day</span><b>${(Number(item.precision_at_k || 0) * 100).toFixed(1)}%</b></div>`).join('');
  $('comparisonTable').innerHTML = `<table><thead><tr><th>Model</th><th>Hit rate</th><th>Precision@K</th><th>Mean hits/day</th><th>ROI</th></tr></thead><tbody>${rows.map(([name, item]) => `<tr><td><strong>${esc(labels[name] || name)}</strong></td><td>${(Number(item.hit_rate_at_k || 0) * 100).toFixed(1)}%</td><td>${(Number(item.precision_at_k || 0) * 100).toFixed(1)}%</td><td>${Number(item.mean_hits || 0).toFixed(2)}</td><td>${(Number(payload.payouts?.[name]?.roi || 0) * 100).toFixed(1)}%</td></tr>`).join('')}</tbody></table>`;
  const best = rows.reduce((left, right) => Number(right[1]?.hit_rate_at_k || 0) > Number(left[1]?.hit_rate_at_k || 0) ? right : left, rows[0]);
  const boosted = rows.find(([name]) => name === 'lightgbm');
  $('perfTrees').textContent = boosted ? `${(Number(boosted[1].hit_rate_at_k || 0) * 100).toFixed(1)}%` : '--';
  $('perfEnsemble').textContent = best ? `${(Number(payload.payouts?.[best[0]]?.roi || 0) * 100).toFixed(1)}%` : '--';
  $('roiMetric').textContent = best ? `${(Number(payload.payouts?.[best[0]]?.roi || 0) * 100).toFixed(1)}%` : '--';
}

function renderHistory() {
  const filter = $('historyFilter')?.value || 'all';
  const data = runs.filter((run) => filter === 'all' || run.model === filter);
  if (!data.length) {
    $('historyTable').innerHTML = '<div class="empty"><div class="empty-mark">↺</div><strong>No session runs</strong><p>Forecasts will appear here after you complete a run.</p></div>';
    return;
  }
  $('historyTable').innerHTML = `<table><thead><tr><th>Run time</th><th>Model</th><th>Target date</th><th>Source</th><th>Ranked output</th></tr></thead><tbody>${data.map((run) => `<tr><td>${esc(run.date)}</td><td><strong>${esc(labels[run.model] || run.model)}</strong></td><td>${esc(run.target)}</td><td>${esc(run.source)}</td><td>${esc(run.numbers)}</td></tr>`).join('')}</tbody></table>`;
}

function formatMoney(value) { return `${Math.round(Number(value || 0)).toLocaleString()} ₫`; }

function renderAnalyticsCharts() {
  const models = latestAnalytics?.models || [];
  renderLineChart('hitsChart', models, 'hits');
  renderLineChart('roiChart', models, 'cumulative_roi', true);
}

function changeChartZoom(scale, anchor = 0.5) {
  const dates = [...new Set((latestAnalytics?.models || []).flatMap((model) => model.series.map((row) => row.date)))].sort();
  if (!dates.length) return;
  const currentEnd = chartViewport.end ?? dates.length - 1;
  const currentLength = currentEnd - chartViewport.start + 1;
  const nextLength = Math.max(Math.min(7, dates.length), Math.min(dates.length, Math.round(currentLength * scale)));
  const center = chartViewport.start + anchor * (currentLength - 1);
  let start = Math.round(center - anchor * (nextLength - 1));
  start = Math.max(0, Math.min(dates.length - nextLength, start));
  chartViewport = { start, end: start + nextLength - 1 };
  renderAnalyticsCharts();
}

function panChart(offset) {
  const dates = [...new Set((latestAnalytics?.models || []).flatMap((model) => model.series.map((row) => row.date)))].sort();
  if (!dates.length) return;
  const end = chartViewport.end ?? dates.length - 1;
  const length = end - chartViewport.start + 1;
  const start = Math.max(0, Math.min(dates.length - length, chartViewport.start + offset));
  chartViewport = { start, end: start + length - 1 };
  renderAnalyticsCharts();
}

function renderLineChart(containerId, models, key, percent = false) {
  const container = $(containerId);
  const allDates = [...new Set(models.flatMap((model) => model.series.map((row) => row.date)))].sort();
  if (!allDates.length) {
    container.innerHTML = '<div class="chart-empty">No settled daily results in this period.</div>';
    return;
  }
  const endIndex = Math.min(chartViewport.end ?? allDates.length - 1, allDates.length - 1);
  const startIndex = Math.min(chartViewport.start, endIndex);
  chartViewport = { start: startIndex, end: endIndex };
  const dates = allDates.slice(startIndex, endIndex + 1);
  const visibleModels = models.map((model) => ({
    ...model,
    series: model.series.filter((row) => row.date >= dates[0] && row.date <= dates.at(-1)),
  }));
  const values = visibleModels.flatMap((model) => model.series.map((row) => Number(row[key] || 0)));
  const minimum = Math.min(0, ...values);
  let maximum = Math.max(0, ...values);
  if (minimum === maximum) maximum = minimum + 1;
  const width = 760;
  const showEveryDate = dates.length <= 31;
  const height = showEveryDate ? 340 : 300;
  const margin = { left: 48, right: 18, top: 18, bottom: showEveryDate ? 78 : 42 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const x = (date) => margin.left + (dates.length === 1 ? innerWidth / 2 : dates.indexOf(date) * innerWidth / (dates.length - 1));
  const y = (value) => margin.top + (maximum - value) * innerHeight / (maximum - minimum);
  const ticks = Array.from({ length: 5 }, (_, index) => minimum + (maximum - minimum) * index / 4);
  const grid = ticks.map((tick) => `<line class="chart-grid-line" x1="${margin.left}" x2="${width - margin.right}" y1="${y(tick)}" y2="${y(tick)}"></line><text class="chart-axis-label" x="${margin.left - 8}" y="${y(tick) + 3}" text-anchor="end">${percent ? `${(tick * 100).toFixed(0)}%` : tick.toFixed(tick % 1 ? 1 : 0)}</text>`).join('');
  const paths = visibleModels.map((model, index) => {
    const points = model.series.map((row) => `${x(row.date)},${y(Number(row[key] || 0))}`).join(' ');
    const dots = model.series.length <= 90 ? model.series.map((row) => `<circle cx="${x(row.date)}" cy="${y(Number(row[key] || 0))}" r="2.4" fill="${chartColors[index % chartColors.length]}"><title>${esc(model.model_name)} · ${row.date}: ${percent ? `${(Number(row[key] || 0) * 100).toFixed(1)}%` : Number(row[key] || 0)}</title></circle>`).join('') : '';
    return `<polyline fill="none" stroke="${chartColors[index % chartColors.length]}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" points="${points}"></polyline>${dots}`;
  }).join('');
  const labelDates = showEveryDate
    ? dates
    : dates.filter((_, index) => index % Math.max(1, Math.ceil(dates.length / 8)) === 0 || index === dates.length - 1);
  const xLabels = labelDates.map((date) => showEveryDate
    ? `<text class="chart-axis-label chart-date-label" x="${x(date)}" y="${height - margin.bottom + 14}" text-anchor="end" transform="rotate(-48 ${x(date)} ${height - margin.bottom + 14})">${date.slice(5)}</text>`
    : `<text class="chart-axis-label" x="${x(date)}" y="${height - 10}" text-anchor="middle">${date.slice(5)}</text>`
  ).join('');
  const zero = minimum < 0 && maximum > 0 ? `<line class="chart-zero-line" x1="${margin.left}" x2="${width - margin.right}" y1="${y(0)}" y2="${y(0)}"></line>` : '';
  const legend = visibleModels.map((model, index) => `<span><i style="background:${chartColors[index % chartColors.length]}"></i>${esc(model.model_name)}</span>`).join('');
  const isFullRange = dates.length === allDates.length;
  const isMinimumRange = dates.length <= Math.min(7, allDates.length);
  container.innerHTML = `<div class="chart-toolbar"><span class="chart-viewport">${dates[0]} to ${dates.at(-1)} · ${dates.length} of ${allDates.length} days</span><div class="chart-actions"><button type="button" data-chart-action="out" aria-label="Zoom out" ${isFullRange ? 'disabled' : ''}>−</button><button type="button" data-chart-action="reset" ${isFullRange ? 'disabled' : ''}>Reset</button><button type="button" data-chart-action="in" aria-label="Zoom in" ${isMinimumRange ? 'disabled' : ''}>+</button></div></div><div class="chart-stage"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${percent ? 'Cumulative ROI' : 'Daily hits'} chart">${grid}${zero}${paths}${xLabels}<line class="chart-crosshair" x1="0" x2="0" y1="${margin.top}" y2="${height - margin.bottom}" hidden></line><rect class="chart-interaction" x="${margin.left}" y="${margin.top}" width="${innerWidth}" height="${innerHeight}" tabindex="0" aria-label="Hover for exact daily values; use mouse wheel to zoom and drag to pan."></rect></svg><div class="chart-tooltip" hidden></div></div><div class="chart-help">Hover for exact values · Mouse wheel or +/− to zoom · Drag to pan</div><div class="chart-legend">${legend}</div>`;

  const svg = container.querySelector('svg');
  const overlay = container.querySelector('.chart-interaction');
  const crosshair = container.querySelector('.chart-crosshair');
  const tooltip = container.querySelector('.chart-tooltip');
  let dragStart = null;
  const pointerRatio = (event) => {
    const bounds = svg.getBoundingClientRect();
    const plotLeft = bounds.left + margin.left / width * bounds.width;
    const plotWidth = innerWidth / width * bounds.width;
    return Math.max(0, Math.min(1, (event.clientX - plotLeft) / plotWidth));
  };
  const showTooltip = (event) => {
    const ratio = pointerRatio(event);
    const index = Math.round(ratio * (dates.length - 1));
    const date = dates[index];
    const xPosition = x(date);
    crosshair.hidden = false;
    crosshair.setAttribute('x1', xPosition);
    crosshair.setAttribute('x2', xPosition);
    const rows = visibleModels.map((model, modelIndex) => {
      const row = model.series.find((item) => item.date === date);
      const value = row ? Number(row[key] || 0) : null;
      const formatted = value === null ? '—' : (percent ? `${(value * 100).toFixed(1)}%` : value.toFixed(value % 1 ? 2 : 0));
      return `<div><span><i style="background:${chartColors[modelIndex % chartColors.length]}"></i>${esc(model.model_name)}</span><strong>${formatted}</strong></div>`;
    }).join('');
    tooltip.innerHTML = `<b>${date}</b>${rows}`;
    tooltip.hidden = false;
    const containerBounds = container.getBoundingClientRect();
    const left = event.clientX - containerBounds.left;
    tooltip.style.left = `${Math.max(8, Math.min(containerBounds.width - 190, left + 14))}px`;
    tooltip.style.top = `${Math.max(42, event.clientY - containerBounds.top - 24)}px`;
  };
  overlay.addEventListener('pointermove', (event) => {
    showTooltip(event);
    if (dragStart) overlay.classList.add('dragging');
  });
  overlay.addEventListener('pointerleave', () => {
    if (!dragStart) {
      tooltip.hidden = true;
      crosshair.hidden = true;
    }
  });
  overlay.addEventListener('wheel', (event) => {
    event.preventDefault();
    changeChartZoom(event.deltaY < 0 ? 0.65 : 1.5, pointerRatio(event));
  }, { passive: false });
  overlay.addEventListener('pointerdown', (event) => {
    dragStart = { x: event.clientX, length: dates.length };
    overlay.setPointerCapture(event.pointerId);
  });
  overlay.addEventListener('pointerup', (event) => {
    if (!dragStart) return;
    const bounds = svg.getBoundingClientRect();
    const pixels = innerWidth / width * bounds.width;
    const offset = Math.round(-(event.clientX - dragStart.x) / pixels * dragStart.length);
    dragStart = null;
    overlay.classList.remove('dragging');
    if (offset) panChart(offset);
  });
  container.querySelectorAll('[data-chart-action]').forEach((button) => button.addEventListener('click', () => {
    const action = button.dataset.chartAction;
    if (action === 'in') changeChartZoom(0.65);
    if (action === 'out') changeChartZoom(1.5);
    if (action === 'reset') {
      chartViewport = { start: 0, end: null };
      renderAnalyticsCharts();
    }
  }));
}

function renderAnalyticsDailyTable(modelId) {
  const model = latestAnalytics?.models.find((item) => item.model_id === modelId);
  if (!model) {
    $('analyticsDailyTable').innerHTML = '<div class="empty"><strong>No model data available</strong></div>';
    return;
  }
  $('analyticsDailyTable').innerHTML = `<table><thead><tr><th>Date</th><th>Hits</th><th>Paid occurrences</th><th>Stake</th><th>Receipts</th><th>Profit</th><th>Daily ROI</th><th>Cumulative ROI</th></tr></thead><tbody>${model.series.map((row) => `<tr><td><strong>${esc(row.date)}</strong></td><td>${row.hits}</td><td>${row.paid_occurrences}</td><td>${formatMoney(row.stake)}</td><td>${formatMoney(row.receipts)}</td><td class="${row.profit >= 0 ? 'profit-positive' : 'profit-negative'}">${formatMoney(row.profit)}</td><td>${(Number(row.roi) * 100).toFixed(1)}%</td><td>${(Number(row.cumulative_roi) * 100).toFixed(1)}%</td></tr>`).join('')}</tbody></table>`;
}

function renderAnalytics(analytics) {
  latestAnalytics = analytics;
  chartViewport = { start: 0, end: null };
  const models = analytics.models || [];
  $('analyticsOutput').hidden = false;
  $('downloadAnalytics').hidden = false;
  $('analyticsPeriodLabel').textContent = analytics.period?.start_date
    ? `${analytics.period.start_date} to ${analytics.period.end_date}` : 'No settled dates';
  $('analyticsSummary').innerHTML = models.map((model, index) => `<article class="analytics-summary-card"><div class="label"><span class="model-swatch" style="background:${chartColors[index % chartColors.length]}"></span>${esc(model.model_name)}</div><div class="value">${(Number(model.summary.roi) * 100).toFixed(1)}% ROI</div><div class="muted">${(Number(model.summary.hit_rate) * 100).toFixed(1)}% hit days · ${Number(model.summary.mean_hits).toFixed(2)} hits/day</div></article>`).join('');
  renderAnalyticsCharts();
  $('analyticsTable').innerHTML = `<table><thead><tr><th>Model</th><th>Days</th><th>Hit rate</th><th>Total hits</th><th>Mean hits/day</th><th>ROI</th><th>Profit</th><th>Max drawdown</th></tr></thead><tbody>${models.map((model) => `<tr><td><strong>${esc(model.model_name)}</strong></td><td>${model.summary.days}</td><td>${(Number(model.summary.hit_rate) * 100).toFixed(1)}%</td><td>${model.summary.total_hits}</td><td>${Number(model.summary.mean_hits).toFixed(2)}</td><td>${(Number(model.summary.roi) * 100).toFixed(1)}%</td><td class="${model.summary.total_profit >= 0 ? 'profit-positive' : 'profit-negative'}">${formatMoney(model.summary.total_profit)}</td><td>${formatMoney(model.summary.max_drawdown)}</td></tr>`).join('')}</tbody></table>`;
  $('analyticsModelFilter').innerHTML = models.map((model) => `<option value="${esc(model.model_id)}">${esc(model.model_name)}</option>`).join('');
  if (models.length) renderAnalyticsDailyTable(models[0].model_id);
}

async function runAnalytics() {
  const status = $('analyticsStatus');
  const button = $('runAnalytics');
  const selectedModels = [...document.querySelectorAll('input[name="analyticsModel"]:checked')].map((input) => input.value);
  const start = $('analyticsStart').value;
  const end = $('analyticsEnd').value;
  const topK = Number($('analyticsTopK').value);
  if (!selectedModels.length) throw new Error('Select at least one model.');
  if (!start || !end || start > end) throw new Error('Choose a valid start and end date.');
  const days = inclusiveDays(start, end);
  if (days < 30 || days > 366) throw new Error('Chart analysis ranges must contain 30 to 366 calendar days.');
  const source = sourceData[0];
  if (source?.first_date && start < source.first_date) throw new Error(`The GitHub feed starts on ${source.first_date}.`);
  if (source?.last_date && end > source.last_date) throw new Error(`Settled chart data ends on ${source.last_date}.`);
  if (!Number.isInteger(topK) || topK < 1 || topK > 20) throw new Error('Ranking size must be between 1 and 20.');
  status.classList.remove('error');
  button.disabled = true;
  setConnection('Computing');
  const envelopes = [];
  try {
    for (let index = 0; index < selectedModels.length; index += 1) {
      const model = selectedModels[index];
      status.textContent = `Running ${labels[model] || model} (${index + 1} of ${selectedModels.length})...`;
      let payload;
      if (model === 'poisson_prize_90d') {
        payload = await fetchPrizeRange({
          source: 'live', model, top_k: topK, test_days: 90,
          target_date: start, target_end_date: end, source_revision: sourceRevision,
        }, (batch, total) => {
          status.textContent = `Running ${labels[model]} · batch ${batch} of ${total}...`;
        });
      } else {
        payload = await postForecast({
          source: 'live', model, top_k: topK, test_days: days,
          target_date: addDays(end, 1), target_end_date: null,
          source_revision: sourceRevision,
        });
      }
      envelopes.push(payload.analytics);
    }
    const analytics = mergeAnalytics(envelopes, selectedModels);
    analytics.models = analytics.models.map((model) => summarizeSeries({
      ...model, series: model.series.filter((row) => row.date >= start && row.date <= end),
    }));
    const missing = selectedModels.filter((modelId) => !analytics.models.some((model) => model.model_id === modelId));
    if (missing.length) throw new Error(`No settled chart data returned for: ${missing.map((id) => labels[id] || id).join(', ')}.`);
    const dates = analytics.models.flatMap((model) => model.series.map((row) => row.date)).sort();
    analytics.period = { start_date: dates[0] || null, end_date: dates.at(-1) || null };
    renderAnalytics(analytics);
    runs.unshift({
      date: new Date().toLocaleString(), model: 'analytics', target: `${start} to ${end}`,
      numbers: `${selectedModels.length} model chart`, source: 'Live GitHub feed',
    });
    runs = runs.slice(0, 20);
    renderHistory();
    status.textContent = `Analysis complete: ${selectedModels.length} model${selectedModels.length === 1 ? '' : 's'} across ${days} days.`;
    setConnection('API ready');
  } finally {
    button.disabled = false;
  }
}

function downloadCsv(payload) {
  const rows = [['Target date', 'Model', 'Rank', 'Number', 'Score', 'Training start', 'Training end']];
  const items = Array.isArray(payload.results) ? payload.results : [payload];
  items.forEach((item) => Object.entries(item.forecasts || {}).forEach(([model, values]) => values.forEach((entry, index) => {
    rows.push([item.target_date, model, index + 1, entry.number, entry.score, item.training_start_date || '', item.training_end_date || '']);
  })));
  const csv = rows.map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(',')).join('\n');
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  const stamp = payload.target_date || `${payload.target_start_date}_${payload.target_end_date}` || 'export';
  link.download = `forecast_${stamp}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function downloadAnalyticsCsv() {
  if (!latestAnalytics) return;
  const headers = ['schema_version', 'model_id', 'model_name', 'date', 'hits', 'hit', 'precision_at_k', 'recall_at_k', 'selections', 'paid_occurrences', 'stake', 'receipts', 'profit', 'roi', 'cumulative_profit', 'cumulative_roi', 'drawdown'];
  const rows = [headers];
  latestAnalytics.models.forEach((model) => model.series.forEach((row) => {
    rows.push(headers.map((header) => {
      if (header === 'schema_version') return latestAnalytics.schema_version;
      if (header === 'model_id') return model.model_id;
      if (header === 'model_name') return model.model_name;
      return row[header];
    }));
  }));
  const csv = rows.map((row) => row.map((value) => `"${String(value ?? '').replaceAll('"', '""')}"`).join(',')).join('\n');
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  link.download = `xsmb_analysis_${latestAnalytics.period.start_date}_${latestAnalytics.period.end_date}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}

document.querySelectorAll('[data-view]').forEach((button) => button.addEventListener('click', () => setPage(button.dataset.view)));
document.querySelectorAll('[data-go]').forEach((button) => button.addEventListener('click', () => setPage(button.dataset.go)));
$('target').addEventListener('input', () => { $('targetEnd').min = $('target').value; });
$('setupTarget').addEventListener('input', () => { $('setupTargetEnd').min = $('setupTarget').value; });
['model', 'setupModel'].forEach((id) => $(id).addEventListener('change', (event) => {
  if (event.target.value === 'poisson_prize_90d') {
    if (id === 'model') $('days').value = '90';
    if (id === 'setupModel') $('setupDays').value = '90';
  }
  syncRangeControls();
}));
$('reset').addEventListener('click', () => {
  $('model').value = 'baseline';
  $('topk').value = '5';
  $('days').value = '30';
  $('targetEnd').value = '';
  syncRangeControls();
  if (sourceData[0]) updateSource(sourceData[0]);
  $('status').textContent = 'Form reset.';
});
$('historyFilter').addEventListener('change', renderHistory);
$('analyticsPeriod').addEventListener('change', updateAnalyticsDates);
$('analyticsModelFilter').addEventListener('change', (event) => renderAnalyticsDailyTable(event.target.value));
$('downloadAnalytics').addEventListener('click', downloadAnalyticsCsv);
$('refreshSource').addEventListener('click', () => {
  refreshGithubSource().catch(() => {});
});
syncRangeControls();

$('form').addEventListener('submit', async (event) => {
  event.preventDefault();
  try { await runForecast(getBody(), 'status'); }
  catch (error) {
    $('resultCards').innerHTML = `<div class="error-box"><strong>Forecast could not be completed.</strong><br>${esc(error.message)}</div>`;
  }
});
$('setupForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  $('model').value = $('setupModel').value;
  $('target').value = $('setupTarget').value;
  $('targetEnd').value = $('setupTargetEnd').value;
  $('topk').value = $('setupTopk').value;
  $('days').value = $('setupDays').value;
  syncRangeControls();
  try { await runForecast(getBody(), 'setupStatus'); }
  catch (error) {
    $('setupStatus').textContent = error.message;
    $('setupStatus').classList.add('error');
  }
});
$('analyticsForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  try { await runAnalytics(); }
  catch (error) {
    $('analyticsStatus').textContent = error.message;
    $('analyticsStatus').classList.add('error');
    setConnection('Needs attention', true);
  }
});
loadSources().catch((error) => {
  $('status').textContent = error.message;
  $('status').classList.add('error');
  setConnection('Offline', true);
  $('dataStatus').textContent = error.message;
});
