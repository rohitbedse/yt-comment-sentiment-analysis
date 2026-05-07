document.addEventListener("DOMContentLoaded", async () => {
  const outputDiv = document.getElementById("output");
  const API_KEY = 'AIzaSyD5jhZpGn9dJC0bB8uqb4DveUVtD8WnHtk';  // Replace with your actual YouTube Data API key 
  const API_URL = 'http://127.0.0.1:5000';

  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;
    const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
    const match = url.match(youtubeRegex);

    if (match && match[1]) {
      const videoId = match[1];
      
      outputDiv.innerHTML = `
        <div class="video-info">
          <div class="video-icon">🎬</div>
          <div class="video-details">
            <div class="video-id">Video ID: ${videoId}</div>
            <div class="video-status">
              <div class="loading-spinner" style="width: 20px; height: 20px; border-width: 3px; display: inline-block; vertical-align: middle; margin-right: 8px;"></div>
              Fetching comments...
            </div>
          </div>
        </div>
      `;

      const comments = await fetchComments(videoId);
      if (comments.length === 0) {
        outputDiv.innerHTML = `
          <div class="info-message">
            <div class="info-icon">💬</div>
            <div>No comments found for this video</div>
          </div>
        `;
        return;
      }

      outputDiv.innerHTML = `
        <div class="video-info">
          <div class="video-icon">✓</div>
          <div class="video-details">
            <div class="video-id">Video ID: ${videoId}</div>
            <div class="video-status">
              <div class="loading-spinner" style="width: 20px; height: 20px; border-width: 3px; display: inline-block; vertical-align: middle; margin-right: 8px;"></div>
              Analyzing ${comments.length} comments...
            </div>
          </div>
        </div>
      `;

      const predictions = await getSentimentPredictions(comments);

      if (predictions) {
        const sentimentCounts = { "1": 0, "0": 0, "-1": 0 };
        const sentimentData = [];
        const totalSentimentScore = predictions.reduce((sum, item) => sum + parseInt(item.sentiment), 0);
        
        predictions.forEach((item, index) => {
          sentimentCounts[item.sentiment]++;
          sentimentData.push({
            timestamp: item.timestamp,
            sentiment: parseInt(item.sentiment)
          });
        });

        const totalComments = comments.length;
        const uniqueCommenters = new Set(comments.map(comment => comment.authorId)).size;
        const totalWords = comments.reduce((sum, comment) => sum + comment.text.split(/\s+/).filter(word => word.length > 0).length, 0);
        const avgWordLength = (totalWords / totalComments).toFixed(1);
        const avgSentimentScore = (totalSentimentScore / totalComments).toFixed(2);
        const normalizedSentimentScore = (((parseFloat(avgSentimentScore) + 1) / 2) * 10).toFixed(1);

        outputDiv.innerHTML = `
          <div class="video-info">
            <div class="video-icon">✓</div>
            <div class="video-details">
              <div class="video-id">Analysis Complete</div>
              <div class="video-status">${totalComments} comments analyzed</div>
            </div>
          </div>

          <div class="stats-grid">
            <div class="stat-card">
              <div class="stat-label">Total Comments</div>
              <div class="stat-value">${totalComments}</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Unique Users</div>
              <div class="stat-value">${uniqueCommenters}</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Avg Length</div>
              <div class="stat-value">${avgWordLength} <span style="font-size: 14px; color: #94a3b8;">words</span></div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Sentiment Score</div>
              <div class="stat-value score">${normalizedSentimentScore}/10</div>
            </div>
          </div>

          <div class="section">
            <div class="section-title">📊 Sentiment Distribution</div>
            <div id="chart-container" class="chart-container"></div>
          </div>

          <div class="section">
            <div class="section-title">📈 Sentiment Over Time</div>
            <div id="trend-graph-container" class="trend-container"></div>
          </div>

          <div class="section">
            <div class="section-title">☁️ Comment Wordcloud</div>
            <div id="wordcloud-container" class="wordcloud-container"></div>
          </div>

          <div class="section">
            <div class="section-title">💬 Top 25 Comments</div>
            <ul class="comment-list">
              ${predictions.slice(0, 25).map((item, index) => {
                const sentimentClass = item.sentiment === "1" ? "sentiment-positive" : 
                                     item.sentiment === "0" ? "sentiment-neutral" : 
                                     "sentiment-negative";
                const sentimentLabel = item.sentiment === "1" ? "Positive" : 
                                     item.sentiment === "0" ? "Neutral" : 
                                     "Negative";
                return `
                  <li class="comment-item">
                    <div>
                      <span class="comment-number">#${index + 1}</span>
                      <span class="comment-sentiment ${sentimentClass}">${sentimentLabel}</span>
                    </div>
                    <div class="comment-text">${item.comment}</div>
                  </li>
                `;
              }).join('')}
            </ul>
          </div>
        `;

        await fetchAndDisplayChart(sentimentCounts);
        await fetchAndDisplayTrendGraph(sentimentData);
        await fetchAndDisplayWordCloud(comments.map(comment => comment.text));
      }
    } else {
      outputDiv.innerHTML = `
        <div class="info-message">
          <div class="info-icon">⚠️</div>
          <div>Please navigate to a YouTube video to analyze comments</div>
        </div>
      `;
    }
  });

  async function fetchComments(videoId) {
    let comments = [];
    let pageToken = "";
    try {
      while (comments.length < 500) {
        const response = await fetch(`https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`);
        const data = await response.json();
        if (data.items) {
          data.items.forEach(item => {
            const commentText = item.snippet.topLevelComment.snippet.textOriginal;
            const timestamp = item.snippet.topLevelComment.snippet.publishedAt;
            const authorId = item.snippet.topLevelComment.snippet.authorChannelId?.value || 'Unknown';
            comments.push({ text: commentText, timestamp: timestamp, authorId: authorId });
          });
        }
        pageToken = data.nextPageToken;
        if (!pageToken) break;
      }
    } catch (error) {
      console.error("Error fetching comments:", error);
      outputDiv.innerHTML = '<div class="error">❌ Error fetching comments</div>';
    }
    return comments;
  }

  async function getSentimentPredictions(comments) {
    try {
      const response = await fetch(`${API_URL}/predict_with_timestamps`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      const result = await response.json();
      if (response.ok) {
        return result;
      } else {
        throw new Error(result.error || 'Error fetching predictions');
      }
    } catch (error) {
      console.error("Error fetching predictions:", error);
      outputDiv.innerHTML = '<div class="error">❌ Error analyzing sentiment</div>';
      return null;
    }
  }

  async function fetchAndDisplayChart(sentimentCounts) {
    try {
      const response = await fetch(`${API_URL}/generate_chart`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_counts: sentimentCounts })
      });
      if (!response.ok) throw new Error('Failed to fetch chart');
      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = imgURL;
      document.getElementById('chart-container').appendChild(img);
    } catch (error) {
      console.error("Error fetching chart:", error);
    }
  }

  async function fetchAndDisplayWordCloud(comments) {
    try {
      const response = await fetch(`${API_URL}/generate_wordcloud`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      if (!response.ok) throw new Error('Failed to fetch wordcloud');
      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = imgURL;
      document.getElementById('wordcloud-container').appendChild(img);
    } catch (error) {
      console.error("Error fetching wordcloud:", error);
    }
  }

  async function fetchAndDisplayTrendGraph(sentimentData) {
    try {
      const response = await fetch(`${API_URL}/generate_trend_graph`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_data: sentimentData })
      });
      if (!response.ok) throw new Error('Failed to fetch trend graph');
      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = imgURL;
      document.getElementById('trend-graph-container').appendChild(img);
    } catch (error) {
      console.error("Error fetching trend graph:", error);
    }
  }
});