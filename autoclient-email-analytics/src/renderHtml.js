const integrationName = "AutoclientAI Email Analytics";
const iconURL = "https://imagedelivery.net/wSMYJvS3Xw-n339CbDyDIA/281012e9-2fdd-4039-69f8-bef12da84400/public";

async function renderHtml(stats, logs) {
  const svg = await getIntegrationSvg();
  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${integrationName}</title>
  <link rel="stylesheet" type="text/css" href="https://templates.cloudflareintegrations.com/styles.css">
  <style>
    .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
    .stats-grid { 
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 20px;
      margin-bottom: 30px;
    }
    .stat-card {
      background-color: #f8f9fa;
      border-radius: 8px;
      padding: 20px;
      text-align: center;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stat-value {
      font-size: 24px;
      font-weight: bold;
      color: #2196F3;
      margin: 10px 0;
    }
    .stat-label {
      color: #666;
      font-size: 14px;
    }
    .logs-container { 
      background-color: #f8f9fa;
      border-radius: 8px;
      padding: 20px;
      margin-top: 20px;
      max-height: 400px;
      overflow-y: auto;
    }
    .log-item { 
      padding: 8px;
      border-bottom: 1px solid #eee;
      font-family: monospace;
    }
    .log-item:last-child {
      border-bottom: none;
    }
    .refresh-button {
      background-color: #2196F3;
      color: white;
      border: none;
      padding: 10px 20px;
      border-radius: 4px;
      cursor: pointer;
      margin-bottom: 20px;
    }
    .refresh-button:hover {
      background-color: #1976D2;
    }
    @media (max-width: 768px) {
      .stats-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
<div class="container">
  <header style="text-align: center; margin-bottom: 40px;">
    ${svg}
    <h1>${integrationName}</h1>
  </header>
  
  <button class="refresh-button" onclick="location.reload()">
    Refresh Dashboard
  </button>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-label">Total Emails Sent</div>
      <div class="stat-value">${stats.totalSent.toLocaleString()}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Total Opens</div>
      <div class="stat-value">${stats.totalOpened.toLocaleString()}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Total Clicks</div>
      <div class="stat-value">${stats.totalClicked.toLocaleString()}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Open Rate</div>
      <div class="stat-value">${stats.openRate}%</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Click Rate</div>
      <div class="stat-value">${stats.clickRate}%</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Avg. Time to Open</div>
      <div class="stat-value">${stats.averageTimeToOpen} min</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Avg. Time to Click</div>
      <div class="stat-value">${stats.averageTimeToClick} min</div>
    </div>
  </div>

  <div class="logs-container">
    <h2>Recent Activity Logs</h2>
    ${logs.map(log => `<div class="log-item">${log}</div>`).join('')}
  </div>
</div>

<script>
  // Auto-refresh every 5 minutes
  setTimeout(() => location.reload(), 300000);
</script>
</body>
</html>
`;
}

async function getIntegrationSvg() {
  try {
    const response = await fetch("https://templates.cloudflareintegrations.com/connection_graphic.svg");
    const svg = await response.text();
    return svg.replace("icon_url", iconURL);
  } catch (error) {
    console.error('Error fetching SVG:', error);
    return ''; // Return empty string if SVG fetch fails
  }
}

export default renderHtml;