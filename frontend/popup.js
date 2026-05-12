document.addEventListener("DOMContentLoaded", async () => {
  const outputDiv = document.getElementById("output");
  const API_KEY = 'AIzaSyD5jhZpGn9dJC0bB8uqb4DveUVtD8WnHtk';  // Replace with your actual YouTube Data API key
  const API_URL = 'http://127.0.0.1:8000';

  // ── IDLE STATE ──────────────────────────────────────────────────────────────
  function showIdle() {
    outputDiv.innerHTML = `
      <div class="idle-state">
        <div class="idle-icon">🎬</div>
        <div class="idle-text">Open a YouTube video<br>to analyze its comments</div>
      </div>`;
  }

  // ── STATUS BAR ───────────────────────────────────────────────────────────────
  function setStatus(type, text, count = '') {
    const dotClass = { loading: '', done: 'done', error: 'err', idle: 'idle' }[type] || '';
    return `
      <div class="status-bar">
        <div class="status-dot ${dotClass}"></div>
        <div class="status-text">${text}</div>
        ${count ? `<div class="status-count">${count}</div>` : ''}
      </div>`;
  }

  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;
    const match = url.match(/^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/);

    if (!match) { showIdle(); return; }

    const videoId = match[1];

    // Step 1: fetching
    outputDiv.innerHTML = setStatus('loading', `Fetching comments for <b>${videoId}</b>…`);

    const comments = await fetchComments(videoId);
    if (!comments || comments.length === 0) {
      outputDiv.innerHTML = `<div class="error-box">❌ No comments found for this video.</div>`;
      return;
    }

    // Step 2: analysing
    outputDiv.innerHTML = setStatus('loading', `Analysing <b>${comments.length}</b> comments…`, `${comments.length} comments`);

    const predictions = await getSentimentPredictions(comments);
    if (!predictions) return;

    // ── CRUNCH NUMBERS ───────────────────────────────────────────────────────
    const buckets = { '1': [], '0': [], '-1': [] };
    const sentimentCounts = { '1': 0, '0': 0, '-1': 0 };
    const sentimentData = [];
    let totalScore = 0;

    predictions.forEach(item => {
      const s = item.sentiment;
      buckets[s].push(item);
      sentimentCounts[s]++;
      totalScore += parseInt(s);
      sentimentData.push({ timestamp: item.timestamp, sentiment: parseInt(s) });
    });

    const total = predictions.length;
    const pctPos = ((sentimentCounts['1']  / total) * 100).toFixed(1);
    const pctNeu = ((sentimentCounts['0']  / total) * 100).toFixed(1);
    const pctNeg = ((sentimentCounts['-1'] / total) * 100).toFixed(1);

    const avgScore = (totalScore / total);
    const normalizedScore = (((avgScore + 1) / 2) * 10).toFixed(1);

    const uniqueUsers = new Set(comments.map(c => c.authorId)).size;
    const totalWords = comments.reduce((s, c) => s + c.text.split(/\s+/).filter(w => w.length > 0).length, 0);
    const avgWords = (totalWords / total).toFixed(0);

    // dominant sentiment
    const dominant = sentimentCounts['1'] >= sentimentCounts['0'] && sentimentCounts['1'] >= sentimentCounts['-1'] ? 'positive'
                   : sentimentCounts['-1'] >= sentimentCounts['0'] ? 'negative'
                   : 'neutral';
    const verdictMeta = {
      positive: { emoji: '😊', label: 'Community Verdict', title: 'Mostly Positive', cls: 'positive' },
      neutral:  { emoji: '😐', label: 'Community Verdict', title: 'Mixed / Neutral',  cls: 'neutral'  },
      negative: { emoji: '😠', label: 'Community Verdict', title: 'Mostly Negative',  cls: 'negative' },
    }[dominant];

    // ── RENDER LAYOUT ────────────────────────────────────────────────────────
    outputDiv.innerHTML = `
      ${setStatus('done', `Analysis complete — ${total} comments`, `${videoId}`)}

      <!-- VERDICT -->
      <div class="verdict-card ${verdictMeta.cls}">
        <div class="verdict-left">
          <div class="verdict-emoji">${verdictMeta.emoji}</div>
          <div>
            <div class="verdict-label">${verdictMeta.label}</div>
            <div class="verdict-title">${verdictMeta.title}</div>
          </div>
        </div>
        <div class="verdict-score">
          <div class="verdict-score-val">${normalizedScore}</div>
          <div class="verdict-score-label">/ 10 score</div>
        </div>
      </div>

      <!-- QUICK STATS -->
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-val">${total}</div>
          <div class="stat-lbl">Comments</div>
        </div>
        <div class="stat-card">
          <div class="stat-val">${uniqueUsers}</div>
          <div class="stat-lbl">Users</div>
        </div>
        <div class="stat-card">
          <div class="stat-val">${avgWords}</div>
          <div class="stat-lbl">Avg Words</div>
        </div>
        <div class="stat-card">
          <div class="stat-val">${pctNeg}%</div>
          <div class="stat-lbl" style="color:var(--neg)">Negative</div>
        </div>
      </div>

      <!-- SENTIMENT BAR -->
      <div class="sentiment-bar-section">
        <div class="section-label">Sentiment Breakdown</div>
        <div class="bar-track">
          <div class="bar-seg pos" style="width:${pctPos}%"></div>
          <div class="bar-seg neu" style="width:${pctNeu}%"></div>
          <div class="bar-seg neg" style="width:${pctNeg}%"></div>
        </div>
        <div class="bar-legend">
          <div class="legend-item">
            <div class="legend-dot pos"></div>
            <span class="legend-text">Positive</span>
            <span class="legend-pct pos">${pctPos}%</span>
          </div>
          <div class="legend-item">
            <div class="legend-dot neu"></div>
            <span class="legend-text">Neutral</span>
            <span class="legend-pct neu">${pctNeu}%</span>
          </div>
          <div class="legend-item">
            <div class="legend-dot neg"></div>
            <span class="legend-text">Negative</span>
            <span class="legend-pct neg">${pctNeg}%</span>
          </div>
        </div>
      </div>

      <!-- CHART -->
      <div class="chart-section">
        <div class="section-label">Distribution Chart</div>
        <div class="chart-wrap" id="chart-container"></div>
      </div>

      <!-- TREND -->
      <div class="chart-section">
        <div class="section-label">Sentiment Over Time</div>
        <div class="chart-wrap" id="trend-graph-container"></div>
      </div>

      <!-- WORD CLOUD -->
      <div class="chart-section">
        <div class="section-label">Word Cloud</div>
        <div class="chart-wrap" id="wordcloud-container"></div>
      </div>

      <!-- TABBED COMMENTS -->
      <div class="tabs-section">
        <div class="tabs-header">
          <button class="tab-btn active pos" data-tab="pos">
            😊 Positive <span class="tab-count">${sentimentCounts['1']}</span>
          </button>
          <button class="tab-btn neu" data-tab="neu">
            😐 Neutral <span class="tab-count">${sentimentCounts['0']}</span>
          </button>
          <button class="tab-btn neg" data-tab="neg">
            😠 Negative <span class="tab-count">${sentimentCounts['-1']}</span>
          </button>
        </div>

        <div class="tab-pane active" id="tab-pos">
          ${renderCommentList(buckets['1'], 'pos', 'Positive')}
        </div>
        <div class="tab-pane" id="tab-neu">
          ${renderCommentList(buckets['0'], 'neu', 'Neutral')}
        </div>
        <div class="tab-pane" id="tab-neg">
          ${renderCommentList(buckets['-1'], 'neg', 'Negative')}
        </div>
      </div>
    `;

    // ── TAB SWITCHING ────────────────────────────────────────────────────────
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(`tab-${tab}`).classList.add('active');
      });
    });

    // ── VISUALS ──────────────────────────────────────────────────────────────
    await fetchAndDisplayChart(sentimentCounts);
    await fetchAndDisplayTrendGraph(sentimentData);
    await fetchAndDisplayWordCloud(comments.map(c => c.text));
  });

  // ── HELPERS ──────────────────────────────────────────────────────────────────

  function renderCommentList(items, cls, label) {
    if (!items || items.length === 0) {
      return `<div class="empty-tab">No ${label.toLowerCase()} comments found</div>`;
    }
    const shown = items.slice(0, 30);
    return `
      <ul class="comment-list">
        ${shown.map((item, i) => `
          <li class="comment-item ${cls}">
            <div class="comment-meta">
              <span class="comment-num">#${i + 1}</span>
              <span class="comment-tag ${cls}">${label}</span>
            </div>
            <div class="comment-text">${escapeHtml(item.comment)}</div>
          </li>
        `).join('')}
      </ul>
      ${items.length > 30 ? `<div style="text-align:center;font-size:11px;color:var(--muted);padding:8px">+${items.length - 30} more</div>` : ''}
    `;
  }

  function escapeHtml(str) {
    return (str || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  async function fetchComments(videoId) {
    let comments = [];
    let pageToken = '';
    try {
      while (comments.length < 500) {
        const res = await fetch(`https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`);
        const data = await res.json();
        if (data.items) {
          data.items.forEach(item => {
            const s = item.snippet.topLevelComment.snippet;
            comments.push({ text: s.textOriginal, timestamp: s.publishedAt, authorId: s.authorChannelId?.value || 'Unknown' });
          });
        }
        pageToken = data.nextPageToken;
        if (!pageToken) break;
      }
    } catch (e) {
      console.error('Error fetching comments:', e);
      outputDiv.innerHTML = `<div class="error-box">❌ Error fetching comments. Check your API key.</div>`;
    }
    return comments;
  }

  async function getSentimentPredictions(comments) {
    try {
      const res = await fetch(`${API_URL}/predict_with_timestamps`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ comments })
      });
      const result = await res.json();
      if (res.ok) return result;
      throw new Error(result.error || 'Prediction failed');
    } catch (e) {
      console.error('Sentiment error:', e);
      outputDiv.innerHTML = `<div class="error-box">❌ Error connecting to analysis server.</div>`;
      return null;
    }
  }

  async function fetchAndDisplayChart(sentimentCounts) {
    try {
      const res = await fetch(`${API_URL}/generate_chart`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sentiment_counts: sentimentCounts })
      });
      if (!res.ok) return;
      const blob = await res.blob();
      const img = new Image();
      img.src = URL.createObjectURL(blob);
      document.getElementById('chart-container').appendChild(img);
    } catch (e) { console.error('Chart error:', e); }
  }

  async function fetchAndDisplayWordCloud(comments) {
    try {
      const res = await fetch(`${API_URL}/generate_wordcloud`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ comments })
      });
      if (!res.ok) return;
      const blob = await res.blob();
      const img = new Image();
      img.src = URL.createObjectURL(blob);
      document.getElementById('wordcloud-container').appendChild(img);
    } catch (e) { console.error('Wordcloud error:', e); }
  }

  async function fetchAndDisplayTrendGraph(sentimentData) {
    try {
      const res = await fetch(`${API_URL}/generate_trend_graph`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sentiment_data: sentimentData })
      });
      if (!res.ok) return;
      const blob = await res.blob();
      const img = new Image();
      img.src = URL.createObjectURL(blob);
      document.getElementById('trend-graph-container').appendChild(img);
    } catch (e) { console.error('Trend error:', e); }
  }
});