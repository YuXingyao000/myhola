/* Verify the statistical census UI through Chrome's native DevTools protocol. */
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {spawn} = require('node:child_process');

async function main() {
  const [url, output, chrome] = process.argv.slice(2);
  assert(url && output && chrome, 'Usage: node tool.cjs URL OUTPUT_DIR CHROME_BINARY');
  const destination = path.resolve(output);
  const profile = path.join(destination, 'browser_profile');
  await fs.mkdir(profile, {recursive: true});
  await fs.rm(path.join(profile, 'DevToolsActivePort'), {force: true});
  const browser = spawn(chrome, ['--headless=new', '--no-sandbox', '--disable-dev-shm-usage',
    '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'], {stdio: 'ignore'});
  let socket;
  const errors = [];
  try {
    let port;
    for (let attempt = 0; attempt < 100; attempt++) {
      try { port = (await fs.readFile(path.join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]; break; }
      catch { await new Promise(resolve => setTimeout(resolve, 100)); }
    }
    assert(port, 'Chrome did not expose DevTools');
    const target = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, {method: 'PUT'})).json();
    socket = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
    let nextId = 1;
    const pending = new Map();
    socket.onmessage = event => {
      const message = JSON.parse(event.data);
      if (message.id) {
        const request = pending.get(message.id);
        if (request) { clearTimeout(request.timer); pending.delete(message.id); message.error ? request.reject(message.error) : request.resolve(message.result); }
      } else if (message.method === 'Runtime.exceptionThrown' ||
                 (message.method === 'Runtime.consoleAPICalled' && message.params.type === 'error')) errors.push(message);
    };
    function call(method, params = {}) {
      return new Promise((resolve, reject) => {
        const id = nextId++;
        const timer = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }, 20000);
        pending.set(id, {resolve, reject, timer});
        socket.send(JSON.stringify({id, method, params}));
      });
    }
    async function evaluate(expression) {
      const result = await call('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
      assert(!result.exceptionDetails, JSON.stringify(result.exceptionDetails));
      return result.result.value;
    }
    await call('Page.enable'); await call('Runtime.enable');
    await call('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false});
    await call('Page.navigate', {url});
    for (let attempt = 0; attempt < 100; attempt++) {
      if (await evaluate('Boolean(window.INTERSECTION_CENSUS_VIEW)')) break;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    assert.equal(await evaluate('window.INTERSECTION_CENSUS_VIEW?.denominator'), 304);
    const checks = [];
    for (const [cohort, count] of Object.entries({main: 304, persistent: 262, regressed: 42, rescued: 107})) {
      for (const threshold of ['0.002', '0.006', '0.01', '0.08']) {
        const result = await evaluate(`(() => {
          for (const [id, value] of ${JSON.stringify([['cohort', cohort], ['threshold', threshold]])}) {
            const element = document.getElementById(id); element.value = value; element.dispatchEvent(new Event('change'));
          }
          const D = window.INTERSECTION_CENSUS_DATA, V = window.INTERSECTION_CENSUS_VIEW;
          const pairs = D.pairs.filter(p => V.ids.includes(p.id));
          const mismatches = [];
          for (const row of document.querySelectorAll('#mechanisms tbody tr')) {
            const key = row.dataset.observation;
            for (const method of ['legacy', 'intersection']) for (const [group, value] of [['positive', true], ['negative', false], ['unknown', null]]) {
              const expected = pairs.filter(p => p[method].flags[V.threshold][key] === value).length;
              const button = row.querySelector('button[data-method="' + method + '"][data-group="' + group + '"]');
              if (!button || Number(button.textContent) !== expected) mismatches.push({key, method, group, expected});
            }
          }
          return {view: V, rows: document.querySelectorAll('#mechanisms tbody tr').length, mismatches};
        })()`);
        assert.equal(result.view.denominator, count); assert.equal(result.mismatches.length, 0);
        checks.push({cohort, threshold, count, observation_rows: result.rows});
      }
    }
    const interactions = await evaluate(`(() => {
      const cohort = document.getElementById('cohort'); cohort.value = 'regressed'; cohort.dispatchEvent(new Event('change'));
      const search = document.getElementById('case-search'); search.value = '00021937'; search.dispatchEvent(new Event('input'));
      const caseRendered = document.querySelectorAll('#case-detail article').length === 2;
      const button = document.querySelector('#mechanisms button[data-method="intersection"][data-group="positive"]:not(:disabled)');
      button.click();
      return {caseRendered, expectedIds: Number(button.textContent), displayedIds: document.querySelectorAll('#selected-ids button').length};
    })()`);
    assert(interactions.caseRendered); assert.equal(interactions.expectedIds, interactions.displayedIds);
    await evaluate(`document.getElementById('cohort').value='main'; document.getElementById('threshold').value='0.006'; document.getElementById('cohort').dispatchEvent(new Event('change')); window.scrollTo(0,0);`);
    async function screenshot(name) { const image = await call('Page.captureScreenshot', {format: 'png'}); await fs.writeFile(path.join(destination, name), Buffer.from(image.data, 'base64')); }
    await screenshot('desktop.png');
    const assets = await evaluate(`Promise.all([...new Set([...document.querySelectorAll('a[href],img[src],script[src]')].map(e=>e.href||e.src).filter(u=>u&&u.startsWith(location.origin)))].map(async url=>({url,status:(await fetch(url,{method:'HEAD'})).status})))`);
    assert(assets.every(asset => asset.status === 200), JSON.stringify(assets.filter(asset => asset.status !== 200)));
    await call('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 1, mobile: true});
    await evaluate('window.scrollTo(0,0)');
    const mobile = await evaluate('({viewport:innerWidth,document:document.documentElement.scrollWidth})');
    assert(mobile.document <= mobile.viewport, JSON.stringify(mobile));
    await screenshot('mobile.png');
    assert.equal(errors.length, 0, JSON.stringify(errors));
    const report = {url, checked_at_utc: new Date().toISOString(), cohort_threshold_checks: checks,
      interactions, assets_checked: assets.length, failed_assets: assets.filter(asset => asset.status !== 200), mobile, javascript_errors: errors};
    await fs.writeFile(path.join(destination, 'browser_check.json'), JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report));
  } finally { if (socket) socket.close(); browser.kill('SIGTERM'); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
