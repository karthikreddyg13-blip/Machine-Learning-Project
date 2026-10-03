/**
 * MATCH OUTCOME PREDICTION SYSTEM
 * Sports Analytics using Logistic Regression and Random Forest
 * Frontend Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const apiStatusDot = document.getElementById('api-status-dot');
  const apiStatusText = document.getElementById('api-status-text');
  
  const teamASelect = document.getElementById('team-a-select');
  const teamBSelect = document.getElementById('team-b-select');
  const venueSelect = document.getElementById('venue-select');
  
  const teamAAvgScore = document.getElementById('team-a-avg-score');
  const teamBAvgScore = document.getElementById('team-b-avg-score');
  const teamAWinRate = document.getElementById('team-a-win-rate');
  const teamBWinRate = document.getElementById('team-b-win-rate');
  const teamAAvgGoals = document.getElementById('team-a-avg-goals');
  const teamBAvgGoals = document.getElementById('team-b-avg-goals');
  const teamAPerf = document.getElementById('team-a-player-performance');
  const teamBPerf = document.getElementById('team-b-player-performance');

  const predictionForm = document.getElementById('prediction-form');
  const predictBtn = document.getElementById('predict-btn');
  const predictSpinner = document.getElementById('predict-spinner');
  const formAlert = document.getElementById('form-alert');

  // Results elements
  const resultsPlaceholder = document.getElementById('results-placeholder');
  const resultsContent = document.getElementById('results-content');
  const loadInitialDemoBtn = document.getElementById('load-initial-demo-btn');
  
  const resTeamAName = document.getElementById('res-team-a-name');
  const resTeamBName = document.getElementById('res-team-b-name');
  const resVenuePill = document.getElementById('res-venue-pill');

  const lrPredictionBadge = document.getElementById('lr-prediction-badge');
  const lrWinnerText = document.getElementById('lr-winner-text');
  const lrWinPct = document.getElementById('lr-win-pct');
  const lrLossPct = document.getElementById('lr-loss-pct');
  const lrWinBar = document.getElementById('lr-win-bar');
  const lrLossBar = document.getElementById('lr-loss-bar');

  const rfPredictionBadge = document.getElementById('rf-prediction-badge');
  const rfWinnerText = document.getElementById('rf-winner-text');
  const rfWinPct = document.getElementById('rf-win-pct');
  const rfLossPct = document.getElementById('rf-loss-pct');
  const rfWinBar = document.getElementById('rf-win-bar');
  const rfLossBar = document.getElementById('rf-loss-bar');

  const consensusBadge = document.getElementById('consensus-badge');
  const summaryLrVal = document.getElementById('summary-lr-val');
  const summaryRfVal = document.getElementById('summary-rf-val');
  const summaryAvgVal = document.getElementById('summary-avg-val');

  // Comparison & Dataset elements
  const comparisonTbody = document.getElementById('comparison-tbody');
  const featuresImportanceList = document.getElementById('features-importance-list');
  const datasetThead = document.getElementById('dataset-thead');
  const datasetTbody = document.getElementById('dataset-tbody');
  const datasetSearch = document.getElementById('dataset-search');
  const tableRowCount = document.getElementById('table-row-count');

  let rawDataset = [];
  let comparisonChartInstance = null;

  // Preset sample matches based on dataset
  const samplePresets = [
    {
      teamA: 'Team A', teamB: 'Team B', venue: 'Home',
      scoreA: 78, scoreB: 72, winA: 65, winB: 58,
      goalsA: 12, goalsB: 8, perfA: 62, perfB: 55
    },
    {
      teamA: 'Team C', teamB: 'Team D', venue: 'Away',
      scoreA: 64, scoreB: 70, winA: 59, winB: 63,
      goalsA: 9, goalsB: 11, perfA: 48, perfB: 52
    },
    {
      teamA: 'Team E', teamB: 'Team F', venue: 'Home',
      scoreA: 82, scoreB: 75, winA: 71, winB: 60,
      goalsA: 14, goalsB: 7, perfA: 68, perfB: 57
    },
    {
      teamA: 'Team G', teamB: 'Team H', venue: 'Neutral',
      scoreA: 69, scoreB: 66, winA: 61, winB: 58,
      goalsA: 10, goalsB: 9, perfA: 55, perfB: 51
    }
  ];

  // Initialize Application
  initApp();

  async function initApp() {
    try {
      await checkHealth();
      await loadTeams();
      await loadVenues();
      await loadDatasetAndComparison();
      setupPresets();
      setupEventListeners();
    } catch (err) {
      console.error('Initialization error:', err);
      showError('Failed to initialize connection to ML backend. Please ensure the server is running.');
    }
  }

  // 1. Health Check
  async function checkHealth() {
    try {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();
      
      if (data.status === 'healthy') {
        apiStatusDot.style.backgroundColor = 'var(--color-win)';
        apiStatusDot.style.boxShadow = '0 0 10px var(--color-win)';
        apiStatusText.textContent = 'Models Loaded & Online';
      }
    } catch (e) {
      apiStatusDot.style.backgroundColor = 'var(--color-loss)';
      apiStatusDot.style.boxShadow = '0 0 10px var(--color-loss)';
      apiStatusText.textContent = 'Backend Offline';
      console.warn('Backend health check failed:', e);
    }
  }

  // 2. Load Teams
  async function loadTeams() {
    const res = await fetch('/api/teams');
    const data = await res.json();
    const teams = data.teams || [];

    teamASelect.innerHTML = '';
    teamBSelect.innerHTML = '';

    teams.forEach((team, idx) => {
      const optA = document.createElement('option');
      optA.value = team;
      optA.textContent = team;
      teamASelect.appendChild(optA);

      const optB = document.createElement('option');
      optB.value = team;
      optB.textContent = team;
      teamBSelect.appendChild(optB);
    });

    // Default select Team A and Team B
    if (teams.length >= 2) {
      teamASelect.value = teams[0];
      teamBSelect.value = teams[1];
    }
  }

  // 3. Load Venues
  async function loadVenues() {
    const res = await fetch('/api/venues');
    const data = await res.json();
    const venues = data.venues || [];

    venueSelect.innerHTML = '';
    venues.forEach((v) => {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = v;
      venueSelect.appendChild(opt);
    });

    if (venues.includes('Home')) {
      venueSelect.value = 'Home';
    }
  }

  // 4. Load Dataset Info & Model Comparison Results
  async function loadDatasetAndComparison() {
    const res = await fetch('/api/dataset-info');
    if (!res.ok) throw new Error('Failed to fetch dataset info');
    const data = await res.json();

    // Populate Overview Stats
    document.getElementById('stat-matches').textContent = data.total_matches + '+';
    document.getElementById('stat-attributes').textContent = data.total_features;
    document.getElementById('dset-total-matches').textContent = data.total_matches;
    document.getElementById('dset-total-features').textContent = data.total_features;
    document.getElementById('dset-num-teams').textContent = data.number_of_teams;
    document.getElementById('dset-num-venues').textContent = data.number_of_venues;

    // Populate Model Comparison Table
    renderComparisonTable(data.model_comparison);

    // Render Comparison Chart
    renderComparisonChart(data.model_comparison);

    // Render Feature Importances
    renderFeatureImportances(data.top_features);

    // Store and Render Dataset Table
    rawDataset = data.sample_data || [];
    renderDatasetTable(data.columns, rawDataset);
  }

  function renderComparisonTable(comparisonData) {
    if (!comparisonData || comparisonData.length === 0) return;
    comparisonTbody.innerHTML = '';

    comparisonData.forEach((row) => {
      const tr = document.createElement('tr');
      const isLR = row.Model.includes('Logistic');

      tr.innerHTML = `
        <td class="metric-highlight">
          <strong>${row.Model}</strong>
          ${isLR ? '<span style="font-size:0.75rem; color:var(--accent-cyan); display:block;">Linear Hyperplane</span>' : '<span style="font-size:0.75rem; color:#a5b4fc; display:block;">Tree Ensemble</span>'}
        </td>
        <td><strong>${parseFloat(row['Accuracy']).toFixed(1)}%</strong></td>
        <td>${parseFloat(row['Precision']).toFixed(1)}%</td>
        <td>${parseFloat(row['Recall']).toFixed(1)}%</td>
        <td><strong>${parseFloat(row['F1 Score']).toFixed(2)}%</strong></td>
      `;
      comparisonTbody.appendChild(tr);
    });
  }

  function renderComparisonChart(comparisonData) {
    if (!comparisonData || comparisonData.length < 2) return;
    const ctx = document.getElementById('modelComparisonChart');
    if (!ctx) return;

    const lrRow = comparisonData.find(r => r.Model.includes('Logistic')) || comparisonData[0];
    const rfRow = comparisonData.find(r => r.Model.includes('Random')) || comparisonData[1];

    const metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score'];
    const lrScores = [
      parseFloat(lrRow['Accuracy']),
      parseFloat(lrRow['Precision']),
      parseFloat(lrRow['Recall']),
      parseFloat(lrRow['F1 Score'])
    ];
    const rfScores = [
      parseFloat(rfRow['Accuracy']),
      parseFloat(rfRow['Precision']),
      parseFloat(rfRow['Recall']),
      parseFloat(rfRow['F1 Score'])
    ];

    if (comparisonChartInstance) {
      comparisonChartInstance.destroy();
    }

    comparisonChartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: metrics,
        datasets: [
          {
            label: 'Logistic Regression',
            data: lrScores,
            backgroundColor: 'rgba(59, 130, 246, 0.75)',
            borderColor: '#3b82f6',
            borderWidth: 1.5,
            borderRadius: 6,
          },
          {
            label: 'Random Forest',
            data: rfScores,
            backgroundColor: 'rgba(6, 182, 212, 0.75)',
            borderColor: '#06b6d4',
            borderWidth: 1.5,
            borderRadius: 6,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: {
              color: '#cbd5e1',
              font: { family: 'Inter', size: 12, weight: 600 }
            }
          },
          tooltip: {
            backgroundColor: 'rgba(15, 23, 42, 0.95)',
            titleColor: '#ffffff',
            bodyColor: '#e2e8f0',
            borderColor: 'rgba(79, 110, 160, 0.3)',
            borderWidth: 1,
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(1)}%`
            }
          }
        },
        scales: {
          y: {
            min: 0,
            max: 105,
            ticks: {
              color: '#94a3b8',
              callback: (v) => v + '%'
            },
            grid: {
              color: 'rgba(255, 255, 255, 0.05)'
            }
          },
          x: {
            ticks: {
              color: '#cbd5e1',
              font: { family: 'Inter', weight: 600 }
            },
            grid: {
              display: false
            }
          }
        }
      }
    });
  }

  function renderFeatureImportances(features) {
    if (!features || features.length === 0) return;
    featuresImportanceList.innerHTML = '';

    features.slice(0, 8).forEach(item => {
      const cleanName = item.feature.replace(/_/g, ' ');
      const barItem = document.createElement('div');
      barItem.className = 'feature-bar-item';
      barItem.innerHTML = `
        <div class="fbar-meta">
          <span class="fbar-name">${cleanName}</span>
          <span class="fbar-val">${item.importance.toFixed(2)}%</span>
        </div>
        <div class="fbar-track">
          <div class="fbar-fill" style="width: ${Math.min(100, item.importance * 3.5)}%;"></div>
        </div>
      `;
      featuresImportanceList.appendChild(barItem);
    });
  }

  function renderDatasetTable(columns, rows) {
    if (!columns || !rows) return;

    // Header
    datasetThead.innerHTML = `
      <tr>
        ${columns.map(col => `<th>${col.replace(/_/g, ' ')}</th>`).join('')}
      </tr>
    `;

    updateTableRows(rows, columns);
  }

  function updateTableRows(rows, columns) {
    datasetTbody.innerHTML = '';
    tableRowCount.textContent = `Showing ${rows.length} of ${rawDataset.length} matches`;

    if (rows.length === 0) {
      datasetTbody.innerHTML = `<tr><td colspan="${columns.length}" class="text-center">No matches found for current search.</td></tr>`;
      return;
    }

    rows.forEach(row => {
      const tr = document.createElement('tr');
      const cells = columns.map(col => {
        let val = row[col];
        if (col === 'Target') {
          return val === 1 
            ? `<td><span class="target-badge win">1 (Win)</span></td>` 
            : `<td><span class="target-badge loss">0 (Loss)</span></td>`;
        }
        if (col === 'Winner') {
          return `<td><strong>${val}</strong></td>`;
        }
        return `<td>${val}</td>`;
      }).join('');
      tr.innerHTML = cells;
      datasetTbody.appendChild(tr);
    });
  }

  // 5. Presets Handling
  function setupPresets() {
    samplePresets.forEach((preset, index) => {
      const btn = document.getElementById(`preset-${index + 1}`);
      if (btn) {
        btn.addEventListener('click', () => applyPreset(preset));
      }
    });

    if (loadInitialDemoBtn) {
      loadInitialDemoBtn.addEventListener('click', () => {
        applyPreset(samplePresets[0]);
        predictionForm.dispatchEvent(new Event('submit'));
      });
    }
  }

  function applyPreset(p) {
    teamASelect.value = p.teamA;
    teamBSelect.value = p.teamB;
    venueSelect.value = p.venue;
    teamAAvgScore.value = p.scoreA;
    teamBAvgScore.value = p.scoreB;
    teamAWinRate.value = p.winA;
    teamBWinRate.value = p.winB;
    teamAAvgGoals.value = p.goalsA;
    teamBAvgGoals.value = p.goalsB;
    teamAPerf.value = p.perfA;
    teamBPerf.value = p.perfB;

    hideError();
  }

  // 6. Event Listeners & Validation
  function setupEventListeners() {
    // Search in Dataset
    if (datasetSearch) {
      datasetSearch.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase().trim();
        if (!query) {
          updateTableRows(rawDataset, Object.keys(rawDataset[0] || {}));
          return;
        }

        const filtered = rawDataset.filter(row => {
          return Object.values(row).some(val => 
            String(val).toLowerCase().includes(query)
          );
        });
        updateTableRows(filtered, Object.keys(rawDataset[0] || {}));
      });
    }

    // Dynamic Team A vs Team B equality check
    teamASelect.addEventListener('change', checkTeamEquality);
    teamBSelect.addEventListener('change', checkTeamEquality);

    // Form Submission
    predictionForm.addEventListener('submit', handlePredictionSubmit);
  }

  function checkTeamEquality() {
    const a = teamASelect.value;
    const b = teamBSelect.value;
    if (a && b && a.trim().toLowerCase() === b.trim().toLowerCase()) {
      showError('Team A and Team B cannot be the same team. Please select distinct opponents.');
      teamASelect.classList.add('input-error');
      teamBSelect.classList.add('input-error');
      return false;
    } else {
      hideError();
      teamASelect.classList.remove('input-error');
      teamBSelect.classList.remove('input-error');
      return true;
    }
  }

  function showError(msg) {
    formAlert.textContent = msg;
    formAlert.className = 'alert-banner error';
    formAlert.classList.remove('hidden');
  }

  function hideError() {
    formAlert.classList.add('hidden');
  }

  async function handlePredictionSubmit(e) {
    e.preventDefault();

    if (!checkTeamEquality()) {
      return;
    }

    // Validate numerical inputs
    const inputs = [
      teamAAvgScore, teamBAvgScore,
      teamAWinRate, teamBWinRate,
      teamAAvgGoals, teamBAvgGoals,
      teamAPerf, teamBPerf
    ];

    for (const inp of inputs) {
      if (inp.value === '' || isNaN(parseFloat(inp.value))) {
        showError(`Please provide a valid numerical value for ${inp.name.replace(/_/g, ' ')}`);
        inp.focus();
        inp.classList.add('input-error');
        return;
      }
      inp.classList.remove('input-error');
    }

    // Prepare payload
    const payload = {
      team_a: teamASelect.value,
      team_b: teamBSelect.value,
      venue: venueSelect.value,
      team_a_avg_score: parseFloat(teamAAvgScore.value),
      team_b_avg_score: parseFloat(teamBAvgScore.value),
      team_a_win_rate: parseFloat(teamAWinRate.value),
      team_b_win_rate: parseFloat(teamBWinRate.value),
      team_a_avg_goals: parseFloat(teamAAvgGoals.value),
      team_b_avg_goals: parseFloat(teamBAvgGoals.value),
      team_a_player_performance: parseFloat(teamAPerf.value),
      team_b_player_performance: parseFloat(teamBPerf.value),
    };

    // UI Loading state
    predictBtn.disabled = true;
    predictSpinner.classList.remove('hidden');
    predictBtn.querySelector('.btn-text').textContent = 'Computing Predictions...';
    hideError();

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Server error (${response.status})`);
      }

      const result = await response.json();
      displayPredictionResults(payload, result);

    } catch (err) {
      console.error('Prediction failed:', err);
      showError(err.message || 'An error occurred while computing the prediction.');
    } finally {
      predictBtn.disabled = false;
      predictSpinner.classList.add('hidden');
      predictBtn.querySelector('.btn-text').textContent = 'Predict Match Outcome';
    }
  }

  function displayPredictionResults(inputData, res) {
    // Show results content, hide placeholder
    resultsPlaceholder.classList.add('hidden');
    resultsContent.classList.remove('hidden');

    // Header Match Summary
    resTeamAName.textContent = inputData.team_a;
    resTeamBName.textContent = inputData.team_b;
    resVenuePill.textContent = `Venue: ${inputData.venue}`;

    const lr = res.logistic_regression;
    const rf = res.random_forest;

    // Helper to format probabilities regardless of whether 0..1 or 0..100
    const formatPct = (val) => {
      let num = typeof val === 'number' ? val : parseFloat(val);
      if (num <= 1 && num > 0) num = num * 100;
      return num.toFixed(2);
    };

    // 1. Logistic Regression Card
    const lrWin = parseFloat(formatPct(lr.team_a_probability));
    const lrLoss = parseFloat(formatPct(lr.team_a_loss_probability));
    const lrPredText = lr.prediction.toUpperCase();
    const lrIsWin = lrPredText.includes('WINS');

    lrPredictionBadge.textContent = lrPredText;
    lrPredictionBadge.className = `outcome-badge ${lrIsWin ? 'win' : 'loss'}`;
    lrWinnerText.textContent = `Predicted Winner: ${lrIsWin ? inputData.team_a : inputData.team_b}`;
    lrWinPct.textContent = `${lrWin}%`;
    lrLossPct.textContent = `${lrLoss}%`;
    lrWinBar.style.width = `${lrWin}%`;
    lrLossBar.style.width = `${lrLoss}%`;

    // 2. Random Forest Card
    const rfWin = parseFloat(formatPct(rf.team_a_probability));
    const rfLoss = parseFloat(formatPct(rf.team_a_loss_probability));
    const rfPredText = rf.prediction.toUpperCase();
    const rfIsWin = rfPredText.includes('WINS');

    rfPredictionBadge.textContent = rfPredText;
    rfPredictionBadge.className = `outcome-badge ${rfIsWin ? 'win' : 'loss'}`;
    rfWinnerText.textContent = `Predicted Winner: ${rfIsWin ? inputData.team_a : inputData.team_b}`;
    rfWinPct.textContent = `${rfWin}%`;
    rfLossPct.textContent = `${rfLoss}%`;
    rfWinBar.style.width = `${rfWin}%`;
    rfLossBar.style.width = `${rfLoss}%`;

    // 3. Consensus & Agreement
    const modelsAgree = lrIsWin === rfIsWin;
    consensusBadge.textContent = modelsAgree 
      ? `CONSENSUS: FULL AGREEMENT (${lrIsWin ? inputData.team_a : inputData.team_b} WINS)`
      : `DIVERGENT PREDICTIONS (LR: ${lrIsWin ? 'Team A' : 'Team B'} vs RF: ${rfIsWin ? 'Team A' : 'Team B'})`;
    consensusBadge.className = `consensus-badge ${modelsAgree ? '' : 'divergent'}`;

    summaryLrVal.textContent = `${lrWin}% Win (${lrIsWin ? 'Wins' : 'Loses'})`;
    summaryRfVal.textContent = `${rfWin}% Win (${rfIsWin ? 'Wins' : 'Loses'})`;
    
    const avgWin = ((lrWin + rfWin) / 2).toFixed(2);
    summaryAvgVal.textContent = `${avgWin}% Avg (${avgWin >= 50 ? inputData.team_a + ' Favored' : inputData.team_b + ' Favored'})`;

    // Smooth scroll to results
    resultsContent.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // Active Nav Link Spy on Scroll
  const navSections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-links .nav-item');

  window.addEventListener('scroll', () => {
    let currentId = '';
    const scrollPos = window.scrollY + 120;

    navSections.forEach(section => {
      const top = section.offsetTop;
      const height = section.offsetHeight;
      if (scrollPos >= top && scrollPos < top + height) {
        currentId = section.getAttribute('id');
      }
    });

    if (currentId) {
      navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${currentId}`) {
          link.classList.add('active');
        }
      });
    }
  });

});
