(function () {
  'use strict';

  const LS_KEY = {
    apiKey: 'rlhf-ta-api-key',
    baseUrl: 'rlhf-ta-base-url',
    model: 'rlhf-ta-model',
    open: 'rlhf-ta-open'
  };

  const DEFAULT_BASE_URL = 'https://api.kimi.com/coding/v1';
  const DEFAULT_MODEL = 'k3.1-pro';

  const MODELS = [
    { value: 'k3.1-pro', label: 'Kimi Code k3.1 pro' },
    { value: 'kimi-for-coding', label: 'Kimi Code' },
    { value: 'k3', label: 'Kimi k3' },
    { value: 'k3-256k', label: 'Kimi k3 256k' }
  ];

  const PROXY_EXAMPLE = `const http = require('http');
const https = require('https');
const url = require('url');

const PROXY_PORT = 8787;
const TARGET = 'https://api.kimi.com';

http.createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'authorization, content-type');
  if (req.method === 'OPTIONS') { res.writeHead(204); return res.end(); }

  const targetUrl = url.parse(TARGET + req.url);
  const proxyReq = https.request({
    hostname: targetUrl.hostname,
    port: targetUrl.port || 443,
    path: targetUrl.path,
    method: req.method,
    headers: { ...req.headers, host: targetUrl.host }
  }, proxyRes => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res);
  });
  proxyReq.on('error', e => { res.writeHead(502); res.end(e.message); });
  req.pipe(proxyReq);
}).listen(PROXY_PORT, () => console.log('Proxy on http://localhost:' + PROXY_PORT));
`;

  function getStore(key, fallback) {
    try { return localStorage.getItem(key) || fallback; } catch (e) { return fallback; }
  }
  function setStore(key, value) {
    try { localStorage.setItem(key, value); } catch (e) {}
  }

  function createWidget() {
    const fab = document.createElement('button');
    fab.id = 'ta-fab';
    fab.title = 'Teaching assistant';
    fab.setAttribute('aria-label', 'Open teaching assistant');
    fab.setAttribute('aria-expanded', 'false');
    fab.innerHTML = '&#129302;'; // robot face

    const panel = document.createElement('div');
    panel.id = 'ta-panel';
    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-label', 'Teaching assistant');

    panel.innerHTML = `
      <div id="ta-panel-header">
        <select id="ta-model" aria-label="Model">
          ${MODELS.map(m => `<option value="${m.value}">${m.label}</option>`).join('')}
          <option value="claude" disabled>Claude — Coming soon</option>
          <option value="codex" disabled>Codex — Coming soon</option>
        </select>
        <button id="ta-settings-toggle" title="Settings">&#9881;</button>
        <button id="ta-clear" title="Clear chat">&#128465;</button>
        <button id="ta-close" title="Close">&#10005;</button>
      </div>
      <div id="ta-messages"></div>
      <div id="ta-quote-area">
        <div class="ta-quote-box">
          <span class="ta-quote-remove" title="Remove quote">&#10005;</span>
          <div class="ta-quote-text"></div>
        </div>
      </div>
      <div id="ta-settings">
        <label>
          Kimi Code API Key
          <input id="ta-api-key" type="password" placeholder="sk-..." autocomplete="off">
        </label>
        <label>
          API base URL
          <input id="ta-base-url" type="text" placeholder="${DEFAULT_BASE_URL}">
        </label>
        <div class="ta-hint">
          Your key is stored only in this browser's localStorage and is sent directly to the base URL you specify (or a local proxy). It never passes through our servers.
        </div>
        <div class="ta-proxy">
          <strong>Browser CORS note</strong>: direct calls to <code>api.kimi.com</code> are blocked by CORS in most browsers. If you see a network/CORS error, run the tiny Node proxy below and set base URL to <code>http://localhost:8787/v1</code>.
          <pre><code>${PROXY_EXAMPLE.replace(/</g, '&lt;')}</code></pre>
        </div>
      </div>
      <div id="ta-input-area">
        <textarea id="ta-input" rows="1" placeholder="Ask a question..."></textarea>
        <button id="ta-send" title="Send">&#10148;</button>
      </div>
    `;

    document.body.appendChild(fab);
    document.body.appendChild(panel);

    const tooltip = document.createElement('div');
    tooltip.id = 'ta-selection-tooltip';
    tooltip.textContent = 'Quote & ask';
    document.body.appendChild(tooltip);

    return { fab, panel, tooltip };
  }

  const { fab, panel, tooltip } = createWidget();
  const messagesEl = document.getElementById('ta-messages');
  const inputEl = document.getElementById('ta-input');
  const sendBtn = document.getElementById('ta-send');
  const modelEl = document.getElementById('ta-model');
  const apiKeyEl = document.getElementById('ta-api-key');
  const baseUrlEl = document.getElementById('ta-base-url');
  const settingsEl = document.getElementById('ta-settings');
  const settingsToggle = document.getElementById('ta-settings-toggle');
  const quoteArea = document.getElementById('ta-quote-area');
  const quoteText = quoteArea.querySelector('.ta-quote-text');
  const quoteRemove = quoteArea.querySelector('.ta-quote-remove');
  const closeBtn = document.getElementById('ta-close');
  const clearBtn = document.getElementById('ta-clear');

  let quote = null;
  let isStreaming = false;

  // Restore settings
  apiKeyEl.value = getStore(LS_KEY.apiKey, '');
  baseUrlEl.value = getStore(LS_KEY.baseUrl, DEFAULT_BASE_URL);
  modelEl.value = getStore(LS_KEY.model, DEFAULT_MODEL);
  if (!MODELS.some(m => m.value === modelEl.value)) modelEl.value = DEFAULT_MODEL;

  function addMessage(role, text, html = false) {
    const msg = document.createElement('div');
    msg.className = 'ta-message ' + role;
    if (html) msg.innerHTML = text;
    else msg.textContent = text;
    messagesEl.appendChild(msg);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return msg;
  }

  function getPageContext() {
    const h1 = document.querySelector('#mdbook-content main h1');
    const title = h1 ? h1.textContent.trim() : document.title;
    return { title, path: window.location.pathname };
  }

  function buildSystemPrompt() {
    const ctx = getPageContext();
    return `You are a teaching assistant for the RLHF course textbook. The reader is currently viewing "${ctx.title}" (${ctx.path}).
Answer concisely, accurately, and at graduate-student depth. Prefer equations and concrete examples. When citing the text, quote relevant passages.`;
  }

  function togglePanel(show) {
    panel.classList.toggle('open', show);
    fab.setAttribute('aria-expanded', String(show));
    setStore(LS_KEY.open, show ? '1' : '0');
  }

  fab.addEventListener('click', () => togglePanel(!panel.classList.contains('open')));
  closeBtn.addEventListener('click', () => togglePanel(false));

  clearBtn.addEventListener('click', () => {
    messagesEl.innerHTML = '';
    addMessage('system-notice', 'Chat cleared.');
  });

  settingsToggle.addEventListener('click', () => {
    settingsEl.classList.toggle('open');
  });

  apiKeyEl.addEventListener('input', () => setStore(LS_KEY.apiKey, apiKeyEl.value.trim()));
  baseUrlEl.addEventListener('input', () => setStore(LS_KEY.baseUrl, baseUrlEl.value.trim() || DEFAULT_BASE_URL));
  modelEl.addEventListener('change', () => setStore(LS_KEY.model, modelEl.value));

  function setQuote(text) {
    quote = text ? text.trim() : null;
    if (quote) {
      quoteText.textContent = quote;
      quoteArea.classList.add('open');
    } else {
      quoteArea.classList.remove('open');
    }
  }

  quoteRemove.addEventListener('click', () => setQuote(null));

  // Selection tooltip
  document.addEventListener('mouseup', (e) => {
    const sel = window.getSelection();
    const text = sel ? sel.toString().trim() : '';
    if (!text) { tooltip.style.display = 'none'; return; }

    const main = document.querySelector('#mdbook-content main');
    if (!main || !main.contains(sel.anchorNode)) { tooltip.style.display = 'none'; return; }

    const range = sel.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    tooltip.style.display = 'block';
    tooltip.style.left = (window.scrollX + rect.left + rect.width / 2) + 'px';
    tooltip.style.top = (window.scrollY + rect.top - 36) + 'px';
    tooltip.style.transform = 'translateX(-50%)';
  });

  tooltip.addEventListener('click', () => {
    const sel = window.getSelection();
    const text = sel ? sel.toString().trim() : '';
    if (!text) return;
    setQuote(text);
    togglePanel(true);
    tooltip.style.display = 'none';
    inputEl.focus();
  });

  document.addEventListener('mousedown', (e) => {
    if (e.target !== tooltip) tooltip.style.display = 'none';
  });

  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  sendBtn.addEventListener('click', sendMessage);

  function autoResize() {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + 'px';
  }
  inputEl.addEventListener('input', autoResize);

  async function sendMessage() {
    if (isStreaming) return;
    const comment = inputEl.value.trim();
    if (!comment && !quote) return;

    const apiKey = apiKeyEl.value.trim();
    const baseUrl = baseUrlEl.value.trim() || DEFAULT_BASE_URL;
    const model = modelEl.value;

    if (!apiKey) {
      addMessage('system-notice', 'Please enter your Kimi Code API Key in Settings first.');
      settingsEl.classList.add('open');
      return;
    }

    let userContent = '';
    if (quote) userContent += '> ' + quote.split('\n').join('\n> ') + '\n\n';
    if (comment) userContent += comment;

    addMessage('user', userContent);
    inputEl.value = '';
    inputEl.style.height = 'auto';

    const ctx = getPageContext();
    const messages = [
      { role: 'system', content: buildSystemPrompt() },
      { role: 'user', content: `Context: chapter "${ctx.title}" (${ctx.path}).\n\n` + userContent }
    ];

    isStreaming = true;
    sendBtn.disabled = true;
    const replyEl = addMessage('assistant', '');

    try {
      const resp = await fetch(baseUrl.replace(/\/$/, '') + '/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer ' + apiKey
        },
        body: JSON.stringify({ model, messages, stream: true })
      });

      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`HTTP ${resp.status}: ${text}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed === 'data: [DONE]') continue;
          if (!trimmed.startsWith('data:')) continue;
          try {
            const json = JSON.parse(trimmed.slice(5));
            const delta = json.choices && json.choices[0] && json.choices[0].delta;
            if (delta && delta.content) {
              replyEl.textContent += delta.content;
              messagesEl.scrollTop = messagesEl.scrollHeight;
            }
          } catch (e) {}
        }
      }
    } catch (err) {
      replyEl.remove();
      const isCors = err.message.includes('NetworkError') || err.message.includes('Failed to fetch') || err.message.includes('CORS');
      addMessage('system-notice', 'Error: ' + err.message + (isCors ? '\n\nThis usually means the browser blocked the request due to CORS. Run the local proxy in Settings and set base URL to http://localhost:8787/v1.' : ''));
    } finally {
      isStreaming = false;
      sendBtn.disabled = false;
    }
  }

  // Restore open state
  if (getStore(LS_KEY.open, '0') === '1') togglePanel(true);
})();
