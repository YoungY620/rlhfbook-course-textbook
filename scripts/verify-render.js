/* Render-verification harness for the built books.
 * Usage: node scripts/verify-render.js [baseUrlOrFilePath]
 * Defaults to the local English build.
 */
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const ROOT = path.resolve(__dirname, '..');
const BASE = process.argv[2] || ('file://' + path.join(ROOT, 'book', 'index.html'));
const OUT = path.join(ROOT, 'verify-out');
fs.mkdirSync(OUT, { recursive: true });

const PAGES = [
  'index.html',
  '04-lec3-rl-math.html',
  '08-lec6-dpo.html',
  '14-lec10-regularization.html',
];

(async () => {
  const browser = await chromium.launch();
  const results = { consoleErrors: [], checks: [] };
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await ctx.newPage();
  page.on('console', m => { if (m.type() === 'error') results.consoleErrors.push(m.text()); });
  page.on('pageerror', e => results.consoleErrors.push(String(e)));

  async function checkPage(url, label) {
    await page.goto(url, { waitUntil: 'load' });
    // Wait for MathJax (with timeout fallback so offline CDN issues don't hang)
    await page.waitForFunction('window.__mathjaxReady === true', { timeout: 30000 }).catch(() => {});
    const res = await page.evaluate(() => {
      const out = { mathJaxNodes: 0, residualDelims: [], imgsBroken: [], fab: false, langSwitch: false };
      out.mathJaxNodes = document.querySelectorAll('.MathJax, mjx-container').length;
      // residual delimiters outside code/pre
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
        acceptNode(n) {
          const p = n.parentElement;
          if (!p || ['CODE', 'PRE', 'SCRIPT', 'STYLE'].includes(p.tagName)) return NodeFilter.FILTER_REJECT;
          if (p.closest('.MathJax, mjx-container, code, pre')) return NodeFilter.FILTER_REJECT;
          return NodeFilter.FILTER_ACCEPT;
        }
      });
      const re = /(\\\[|\\\]|\\\(|\\\)|\$\$|\\begin\{)/;
      let n;
      while ((n = walker.nextNode())) {
        if (re.test(n.textContent)) out.residualDelims.push(n.textContent.slice(0, 80));
      }
      document.querySelectorAll('img').forEach(img => {
        if (!(img.complete && img.naturalWidth > 0)) out.imgsBroken.push(img.getAttribute('src'));
      });
      out.fab = !!document.querySelector('#ta-fab');
      out.langSwitch = !!document.querySelector('#lang-switch');
      return out;
    });
    results.checks.push({ page: label, ...res });
    await page.screenshot({ path: path.join(OUT, label + '.png'), fullPage: false });
  }

  // English pages
  for (const p of PAGES) {
    await checkPage(BASE.replace(/index\.html$/, '') + p, 'en-' + p.replace('.html', ''));
  }

  // Chinese home
  const zhBase = BASE.replace(/book\/index\.html$/, 'book/zh/index.html');
  await checkPage(zhBase, 'zh-index');

  // Interaction: language switch (on EN index)
  await page.goto(BASE, { waitUntil: 'load' });
  await Promise.all([
    page.waitForNavigation({ waitUntil: 'load', timeout: 15000 }).catch(() => {}),
    page.evaluate(() => document.getElementById('lang-switch').click()),
  ]);
  results.langSwitchLandsOnZh = page.url().includes('/zh/');

  // Interaction: search
  await page.goto(BASE, { waitUntil: 'load' });
  await page.click('#mdbook-search-toggle');
  await page.fill('#mdbook-searchbar', 'DPO');
  await page.waitForTimeout(4000);
  results.searchResults = await page.evaluate(() =>
    document.querySelectorAll('#mdbook-searchresults li').length);

  // Interaction: TA widget — select text, quote, comment, send (fake key)
  await page.goto(BASE, { waitUntil: 'load' });
  results.taFlow = await page.evaluate(async () => {
    const main = document.querySelector('#mdbook-content main');
    const para = main.querySelector('p');
    const range = document.createRange();
    range.selectNodeContents(para);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    document.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
    const tooltip = document.getElementById('ta-selection-tooltip');
    const tooltipVisible = tooltip && getComputedStyle(tooltip).display !== 'none';
    if (tooltipVisible) tooltip.click();
    const panelOpen = document.getElementById('ta-panel').classList.contains('open');
    const quoteOpen = document.getElementById('ta-quote-area').classList.contains('open');
    // fill a fake key and send to verify request shape & error path
    document.getElementById('ta-api-key').value = 'sk-fake';
    document.getElementById('ta-api-key').dispatchEvent(new Event('input', { bubbles: true }));
    document.getElementById('ta-input').value = 'explain this';
    document.getElementById('ta-send').click();
    await new Promise(r => setTimeout(r, 3000));
    const msgs = [...document.querySelectorAll('.ta-message')].map(m => m.className + ': ' + m.textContent.slice(0, 60));
    return { tooltipVisible, panelOpen, quoteOpen, msgs };
  });

  // Mobile viewport
  const mctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true });
  const mp = await mctx.newPage();
  await mp.goto(BASE, { waitUntil: 'load' });
  await mp.waitForFunction('window.__mathjaxReady === true', { timeout: 30000 }).catch(() => {});
  const overflow = await mp.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 2);
  results.mobileOverflow = overflow;
  await mp.screenshot({ path: path.join(OUT, 'mobile-index.png') });

  await browser.close();
  fs.writeFileSync(path.join(OUT, 'results.json'), JSON.stringify(results, null, 2));
  console.log(JSON.stringify(results, null, 2));
})();
