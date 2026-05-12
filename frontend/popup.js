document.addEventListener("DOMContentLoaded", async () => {
  const outputDiv = document.getElementById("output");
  const API_KEY = '#';
  const API_URL = 'http://127.0.0.1:8000';

  function showIdle() {
    outputDiv.innerHTML = `<div class="idle-state"><div class="idle-icon">🎬</div><div class="idle-text">Open a YouTube video<br>to analyze its comments</div></div>`;
  }

  function setStatus(type, text, count = '') {
    const dotClass = { loading: '', done: 'done', error: 'err', idle: 'idle' }[type] || '';
    return `<div class="status-bar"><div class="status-dot ${dotClass}"></div><div class="status-text">${text}</div>${count ? `<div class="status-count">${count}</div>` : ''}</div>`;
  }

  function escapeHtml(str) {
    return (str || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  function renderCommentList(items, cls, label) {
    if (!items || items.length === 0) return `<div class="empty-tab">No ${label.toLowerCase()} comments found</div>`;
    const shown = items.slice(0, 30);
    return `<ul class="comment-list">${shown.map((item, i) => `<li class="comment-item ${cls}"><div class="comment-meta"><span class="comment-num">#${i+1}</span><span class="comment-tag ${cls}">${label}</span></div><div class="comment-text">${escapeHtml(item.comment)}</div></li>`).join('')}</ul>${items.length > 30 ? `<div style="text-align:center;font-size:10px;color:var(--muted);padding:6px">+${items.length-30} more</div>` : ''}`;
  }

  async function fetchComments(videoId) {
    let comments = []; let pageToken = '';
    try {
      while (comments.length < 500) {
        const res = await fetch(`https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`);
        const data = await res.json();
        if (data.items) {
          data.items.forEach(item => {
            const s = item.snippet.topLevelComment.snippet;
            comments.push({ text: s.textOriginal, timestamp: s.publishedAt, authorId: s.authorChannelId?.value || 'Unknown', authorName: s.authorDisplayName || 'Anonymous' });
          });
        }
        pageToken = data.nextPageToken;
        if (!pageToken) break;
      }
    } catch (e) {
      console.error('Fetch error:', e);
      outputDiv.innerHTML = `<div class="error-box">❌ Error fetching comments. Check your API key.</div>`;
    }
    return comments;
  }

  async function getSentimentPredictions(comments) {
    try {
      const res = await fetch(`${API_URL}/predict_with_timestamps`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ comments }) });
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
      const res = await fetch(`${API_URL}/generate_chart`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ sentiment_counts: sentimentCounts }) });
      if (!res.ok) return;
      const blob = await res.blob();
      const img = new Image(); img.src = URL.createObjectURL(blob);
      document.getElementById('chart-container')?.appendChild(img);
    } catch (e) { console.error('Chart error:', e); }
  }

  async function fetchAndDisplayWordCloud(comments) {
    try {
      const res = await fetch(`${API_URL}/generate_wordcloud`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ comments }) });
      if (!res.ok) return;
      const blob = await res.blob();
      const img = new Image(); img.src = URL.createObjectURL(blob);
      document.getElementById('wordcloud-container')?.appendChild(img);
    } catch (e) { console.error('Wordcloud error:', e); }
  }

  async function fetchAndDisplayTrendGraph(sentimentData) {
    try {
      const res = await fetch(`${API_URL}/generate_trend_graph`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ sentiment_data: sentimentData }) });
      if (!res.ok) return;
      const blob = await res.blob();
      const img = new Image(); img.src = URL.createObjectURL(blob);
      document.getElementById('trend-graph-container')?.appendChild(img);
    } catch (e) { console.error('Trend error:', e); }
  }

  // ── CREATOR ANALYTICS LOADERS ──
  async function loadFeature(endpoint, body) {
    try {
      const res = await fetch(`${API_URL}${endpoint}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      if (res.ok) return await res.json();
    } catch (e) { console.error(`${endpoint} error:`, e); }
    return null;
  }

  function renderQuestions(data) {
    if (!data || data.total === 0) return '<div style="color:var(--muted);font-size:10px">No questions found</div>';
    let html = `<div class="fp-label" style="color:var(--cyan)">${data.total} questions found</div>`;
    const cats = data.categories || {};
    Object.entries(cats).forEach(([k,v]) => { html += `<span class="fp-tag green">${k} (${v.count})</span> `; });
    html += '<div style="margin-top:6px">';
    (data.questions || []).slice(0,5).forEach(q => { html += `<div class="question-item">${escapeHtml(q)}</div>`; });
    html += '</div>';
    return html;
  }

  function renderControversy(data) {
    if (!data) return '';
    const color = data.score >= 6 ? 'var(--neg)' : data.score >= 3 ? 'var(--orange)' : 'var(--pos)';
    return `<div class="fp-label">Score: <span style="color:${color}">${data.score} / 10</span> — ${data.level}</div>
      <div style="font-size:9px;color:var(--sub);margin-top:2px">Positive ${data.positive_pct}% vs Negative ${data.negative_pct}%</div>
      <div class="controversy-bar"><div class="controversy-seg pos" style="width:${data.positive_pct}%"></div><div class="controversy-seg neu" style="width:${data.neutral_pct}%"></div><div class="controversy-seg neg" style="width:${data.negative_pct}%"></div></div>`;
  }

  function renderTopFans(data) {
    if (!data || !data.fans || data.fans.length === 0) return '<div style="color:var(--muted);font-size:10px">Not enough data</div>';
    return data.fans.slice(0,4).map(f => `<div class="fan-item"><span class="fan-name">@${escapeHtml(f.author)}</span><span class="fan-stats">${f.comment_count} comments, ${f.positive_rate}% positive</span></div>`).join('');
  }

  function renderClusters(data) {
    if (!data || !data.clusters) return '<div style="color:var(--muted);font-size:10px">No complaints found</div>';
    return Object.entries(data.clusters).slice(0,5).map(([k,v]) => `<div class="cluster-item"><span class="cluster-name">${k}</span><span class="cluster-count">${v.count}</span></div>`).join('');
  }

  function renderIdeas(data) {
    if (!data || !data.ideas) return '<div style="color:var(--muted);font-size:10px">No ideas generated</div>';
    let badge = data.powered_by === 'gemini' ? '<span class="gemini-badge">Gemini</span>' : '';
    return badge + data.ideas.slice(0,4).map(i => `<div class="idea-item"><div class="idea-title">"${escapeHtml(i.title)}"</div><div class="idea-reason">${escapeHtml(i.reason)}</div></div>`).join('');
  }

  function renderHypeMoments(data) {
    if (!data || !data.moments || data.moments.length === 0) return '<div style="color:var(--muted);font-size:10px">No timestamps found</div>';
    return data.moments.slice(0,5).map(m => `<div class="hype-item"><span class="hype-ts">${m.timestamp}</span><span class="hype-info">(mentioned ${m.mention_count}×, ${m.positive_ratio}% positive)</span></div>`).join('');
  }

  function renderSmartReplies(data) {
    if (!data || !data.replies || data.replies.length === 0) return '<div style="color:var(--muted);font-size:10px">No replies generated</div>';
    let badge = data.powered_by === 'gemini' ? '<span class="gemini-badge">Gemini</span>' : '';
    return badge + data.replies.slice(0,3).map(r => `<div class="reply-item"><div class="reply-comment">💬 "${escapeHtml((r.comment||'').slice(0,80))}"</div><div class="reply-draft">📝 ${escapeHtml(r.reply)}</div></div>`).join('');
  }

  function renderHealth(data) {
    if (!data) return '';
    const color = data.score >= 75 ? 'var(--pos)' : data.score >= 50 ? 'var(--orange)' : 'var(--neg)';
    return `<div class="fp-label">Health: <span style="color:${color}">${data.score} / 100</span> — ${data.label}</div>
      <div class="health-meter"><div class="health-fill" style="width:${data.score}%;background:${color}"></div></div>
      <div style="font-size:9px;color:var(--muted);margin-top:3px">Toxicity ${data.toxicity} · Questions ${data.question_ratio}% · Returning ${data.returning_fans}</div>`;
  }

  // ══════ MAIN ══════
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;
    const match = url.match(/^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/);
    if (!match) { showIdle(); return; }

    const videoId = match[1];
    outputDiv.innerHTML = setStatus('loading', `Fetching comments for <b>${videoId}</b>…`);

    const comments = await fetchComments(videoId);
    if (!comments || comments.length === 0) {
      outputDiv.innerHTML = `<div class="error-box">❌ No comments found for this video.</div>`;
      return;
    }

    outputDiv.innerHTML = setStatus('loading', `Analysing <b>${comments.length}</b> comments…`, `${comments.length} comments`);
    const predictions = await getSentimentPredictions(comments);
    if (!predictions) return;

    // ── CRUNCH NUMBERS ──
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
    const pctPos = ((sentimentCounts['1'] / total) * 100).toFixed(1);
    const pctNeu = ((sentimentCounts['0'] / total) * 100).toFixed(1);
    const pctNeg = ((sentimentCounts['-1'] / total) * 100).toFixed(1);
    const avgScore = totalScore / total;
    const normalizedScore = (((avgScore + 1) / 2) * 10).toFixed(1);
    const uniqueUsers = new Set(comments.map(c => c.authorId)).size;
    const totalWords = comments.reduce((s, c) => s + c.text.split(/\s+/).filter(w => w.length > 0).length, 0);
    const avgWords = (totalWords / total).toFixed(0);

    const dominant = sentimentCounts['1'] >= sentimentCounts['0'] && sentimentCounts['1'] >= sentimentCounts['-1'] ? 'positive'
                   : sentimentCounts['-1'] >= sentimentCounts['0'] ? 'negative' : 'neutral';
    const verdictMeta = {
      positive: { emoji: '😊', label: 'Community Verdict', title: 'Mostly Positive', cls: 'positive' },
      neutral:  { emoji: '😐', label: 'Community Verdict', title: 'Mixed / Neutral', cls: 'neutral' },
      negative: { emoji: '😠', label: 'Community Verdict', title: 'Mostly Negative', cls: 'negative' },
    }[dominant];

    // ── RENDER ──
    outputDiv.innerHTML = `
      ${setStatus('done', `Analysis complete — ${total} comments`, `${videoId}`)}
      <div class="verdict-card ${verdictMeta.cls}">
        <div class="verdict-left"><div class="verdict-emoji">${verdictMeta.emoji}</div><div><div class="verdict-label">${verdictMeta.label}</div><div class="verdict-title">${verdictMeta.title}</div></div></div>
        <div class="verdict-score"><div class="verdict-score-val">${normalizedScore}</div><div class="verdict-score-label">/ 10 score</div></div>
      </div>
      <div class="stats-grid">
        <div class="stat-card"><div class="stat-val">${total}</div><div class="stat-lbl">Comments</div></div>
        <div class="stat-card"><div class="stat-val">${uniqueUsers}</div><div class="stat-lbl">Users</div></div>
        <div class="stat-card"><div class="stat-val">${avgWords}</div><div class="stat-lbl">Avg Words</div></div>
        <div class="stat-card"><div class="stat-val">${pctNeg}%</div><div class="stat-lbl" style="color:var(--neg)">Negative</div></div>
      </div>
      <div class="sentiment-bar-section">
        <div class="section-label">Sentiment Breakdown</div>
        <div class="bar-track"><div class="bar-seg pos" style="width:${pctPos}%"></div><div class="bar-seg neu" style="width:${pctNeu}%"></div><div class="bar-seg neg" style="width:${pctNeg}%"></div></div>
        <div class="bar-legend">
          <div class="legend-item"><div class="legend-dot pos"></div><span class="legend-text">Positive</span><span class="legend-pct pos">${pctPos}%</span></div>
          <div class="legend-item"><div class="legend-dot neu"></div><span class="legend-text">Neutral</span><span class="legend-pct neu">${pctNeu}%</span></div>
          <div class="legend-item"><div class="legend-dot neg"></div><span class="legend-text">Negative</span><span class="legend-pct neg">${pctNeg}%</span></div>
        </div>
      </div>
      <div class="chart-section"><div class="section-label">Distribution Chart</div><div class="chart-wrap" id="chart-container"></div></div>
      <div class="chart-section"><div class="section-label">Sentiment Over Time</div><div class="chart-wrap" id="trend-graph-container"></div></div>
      <div class="chart-section"><div class="section-label">Word Cloud</div><div class="chart-wrap" id="wordcloud-container"></div></div>
      <div class="tabs-section">
        <div class="tabs-header">
          <button class="tab-btn active pos" data-tab="pos">😊 Positive <span class="tab-count">${sentimentCounts['1']}</span></button>
          <button class="tab-btn neu" data-tab="neu">😐 Neutral <span class="tab-count">${sentimentCounts['0']}</span></button>
          <button class="tab-btn neg" data-tab="neg">😠 Negative <span class="tab-count">${sentimentCounts['-1']}</span></button>
        </div>
        <div class="tab-pane active" id="tab-pos">${renderCommentList(buckets['1'], 'pos', 'Positive')}</div>
        <div class="tab-pane" id="tab-neu">${renderCommentList(buckets['0'], 'neu', 'Neutral')}</div>
        <div class="tab-pane" id="tab-neg">${renderCommentList(buckets['-1'], 'neg', 'Negative')}</div>
      </div>

      <!-- ═══ CREATOR ANALYTICS ═══ -->
      <div class="creator-divider"><span class="creator-divider-text">🚀 Creator Analytics</span></div>

      <div class="feature-grid">
        <div class="feature-card cyan" id="card-questions">
          <div class="feature-header"><div class="feature-icon cyan">❓</div><span class="feature-badge badge-unique">Unique</span></div>
          <div class="feature-num">Feature 01</div>
          <div class="feature-title">Question Extractor</div>
          <div class="feature-desc">Auto-pulls all questions from comments</div>
          <div class="feature-preview" id="preview-questions"><div class="feature-loading"></div><div class="feature-loading" style="width:70%"></div></div>
        </div>
        <div class="feature-card orange" id="card-controversy">
          <div class="feature-header"><div class="feature-icon orange">🔥</div><span class="feature-badge badge-unique">Unique</span></div>
          <div class="feature-num">Feature 02</div>
          <div class="feature-title">Controversy Score</div>
          <div class="feature-desc">How divided is your audience</div>
          <div class="feature-preview" id="preview-controversy"><div class="feature-loading"></div><div class="feature-loading" style="width:60%"></div></div>
        </div>
        <div class="feature-card green" id="card-fans">
          <div class="feature-header"><div class="feature-icon green">⭐</div><span class="feature-badge badge-new">New</span></div>
          <div class="feature-num">Feature 03</div>
          <div class="feature-title">Top Fan Detector</div>
          <div class="feature-desc">Ranks commenters by engagement</div>
          <div class="feature-preview" id="preview-fans"><div class="feature-loading"></div><div class="feature-loading" style="width:80%"></div></div>
        </div>
        <div class="feature-card red" id="card-clusters">
          <div class="feature-header"><div class="feature-icon red">📋</div><span class="feature-badge badge-unique">Unique</span></div>
          <div class="feature-num">Feature 04</div>
          <div class="feature-title">Complaint Clusters</div>
          <div class="feature-desc">Groups negative comments by topic</div>
          <div class="feature-preview" id="preview-clusters"><div class="feature-loading"></div><div class="feature-loading" style="width:65%"></div></div>
        </div>
        <div class="feature-card accent" id="card-ideas">
          <div class="feature-header"><div class="feature-icon accent">💡</div><span class="feature-badge badge-ai">✦ Gemini</span></div>
          <div class="feature-num">Feature 05</div>
          <div class="feature-title">Content Idea Miner</div>
          <div class="feature-desc">AI suggests next video ideas</div>
          <div class="feature-preview" id="preview-ideas"><div class="feature-loading"></div><div class="feature-loading" style="width:75%"></div></div>
        </div>
        <div class="feature-card blue" id="card-hype">
          <div class="feature-header"><div class="feature-icon blue">⏱️</div><span class="feature-badge badge-new">New</span></div>
          <div class="feature-num">Feature 06</div>
          <div class="feature-title">Hype Moment Finder</div>
          <div class="feature-desc">Finds most-mentioned timestamps</div>
          <div class="feature-preview" id="preview-hype"><div class="feature-loading"></div><div class="feature-loading" style="width:55%"></div></div>
        </div>
        <div class="feature-card purple" id="card-replies">
          <div class="feature-header"><div class="feature-icon purple">💬</div><span class="feature-badge badge-ai">✦ Gemini</span></div>
          <div class="feature-num">Feature 07</div>
          <div class="feature-title">Smart Reply Drafts</div>
          <div class="feature-desc">AI-generated reply suggestions</div>
          <div class="feature-preview" id="preview-replies"><div class="feature-loading"></div><div class="feature-loading" style="width:70%"></div></div>
        </div>
        <div class="feature-card pink" id="card-health">
          <div class="feature-header"><div class="feature-icon pink">💊</div><span class="feature-badge badge-core">Core</span></div>
          <div class="feature-num">Feature 08</div>
          <div class="feature-title">Community Health</div>
          <div class="feature-desc">One score to track over time</div>
          <div class="feature-preview" id="preview-health"><div class="feature-loading"></div><div class="feature-loading" style="width:60%"></div></div>
        </div>
      </div>
    `;

    // ── TAB SWITCHING ──
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(`tab-${tab}`).classList.add('active');
      });
    });

    // ── VISUALS ──
    await fetchAndDisplayChart(sentimentCounts);
    await fetchAndDisplayTrendGraph(sentimentData);
    await fetchAndDisplayWordCloud(comments.map(c => c.text));

    // ── LOAD CREATOR ANALYTICS (parallel, non-blocking) ──
    const predBody = { predictions: predictions };
    const comBody = { comments: comments };
    const bothBody = { comments: comments, predictions: predictions };

    // Rule-based features (fast, parallel)
    loadFeature('/extract_questions', comBody).then(d => { if (d) document.getElementById('preview-questions').innerHTML = renderQuestions(d); });
    loadFeature('/controversy_score', predBody).then(d => { if (d) document.getElementById('preview-controversy').innerHTML = renderControversy(d); });
    loadFeature('/top_fans', bothBody).then(d => { if (d) document.getElementById('preview-fans').innerHTML = renderTopFans(d); });
    loadFeature('/complaint_clusters', predBody).then(d => { if (d) document.getElementById('preview-clusters').innerHTML = renderClusters(d); });
    loadFeature('/hype_moments', predBody).then(d => { if (d) document.getElementById('preview-hype').innerHTML = renderHypeMoments(d); });
    loadFeature('/community_health', bothBody).then(d => { if (d) document.getElementById('preview-health').innerHTML = renderHealth(d); });

    // Gemini-powered features (may take longer)
    loadFeature('/content_ideas', comBody).then(d => { if (d) document.getElementById('preview-ideas').innerHTML = renderIdeas(d); });
    loadFeature('/smart_replies', predBody).then(d => { if (d) document.getElementById('preview-replies').innerHTML = renderSmartReplies(d); });
  });
});