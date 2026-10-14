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
  const externalRequests = [];
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
      } else if (message.method === 'Network.requestWillBeSent' && /^https?:/.test(message.params.request.url)) externalRequests.push(message.params.request.url);
      else if (message.method === 'Runtime.exceptionThrown' ||
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
    await call('Page.enable'); await call('Runtime.enable'); await call('Network.enable');
    await call('Network.setBlockedURLs', {urls: ['http://*', 'https://*']});
    await call('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false});
    await call('Page.navigate', {url});
    for (let attempt = 0; attempt < 100; attempt++) {
      if (await evaluate('Boolean(window.PAIRED_CENSUS_VIEW)')) break;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    const identity = await evaluate('({methods: window.PAIRED_CENSUS_DATA.methods, labels: window.PAIRED_CENSUS_DATA.method_labels})');
    if (identity.methods) {
      assert.equal(identity.methods.baseline, 'intersection');
      assert(['local', 'conditioned'].includes(identity.methods.candidate));
      if (identity.methods.candidate === 'conditioned') assert.equal(identity.labels.candidate, '曲线条件调制');
    }
    const cohortCounts = await evaluate('({main: window.PAIRED_CENSUS_DATA.main.denominator, ...window.PAIRED_CENSUS_DATA.cohort_counts})');
    assert.equal(await evaluate('window.PAIRED_CENSUS_VIEW?.denominator'), cohortCounts.main);
    const checks = [];
    for (const [cohort, count] of Object.entries(cohortCounts)) {
      for (const threshold of ['0.002', '0.006', '0.01', '0.08']) {
        const result = await evaluate(`(() => {
          for (const [id, value] of ${JSON.stringify([['cohort', cohort], ['threshold', threshold]])}) {
            const element = document.getElementById(id); element.value = value; element.dispatchEvent(new Event('change'));
          }
          const D = window.PAIRED_CENSUS_DATA, V = window.PAIRED_CENSUS_VIEW;
          const pairs = D.pairs.filter(p => V.ids.includes(p.id));
          const mismatches = [];
          for (const row of document.querySelectorAll('#mechanisms tbody tr')) {
            const key = row.dataset.observation;
            for (const method of ['baseline', 'candidate']) for (const [group, value] of [['positive', true], ['negative', false], ['unknown', null]]) {
              const expected = pairs.filter(p => p[method].flags[V.threshold][key] === value).length;
              const button = row.querySelector('button[data-method="' + method + '"][data-group="' + group + '"]');
              if (!button || Number(button.textContent) !== expected) mismatches.push({key, method, group, expected});
            }
            const changes = {present_to_absent:0, absent_to_present:0, unknown:0};
            for (const pair of pairs) {
              const a=pair.baseline.flags[V.threshold][key], b=pair.candidate.flags[V.threshold][key];
              if(typeof a!=="boolean"||typeof b!=="boolean") changes.unknown++;
              else if(a&&!b) changes.present_to_absent++; else if(!a&&b) changes.absent_to_present++;
            }
            for(const [group,expected] of Object.entries(changes)) {
              const button=row.querySelector('button[data-method="paired"][data-group="'+group+'"]');
              if(!button||Number(button.textContent)!==expected) mismatches.push({key,group,expected});
            }
          }
          return {view: V, rows: document.querySelectorAll('#mechanisms tbody tr').length, mismatches};
        })()`);
        assert.equal(result.view.denominator, count); assert.equal(result.mismatches.length, 0);
        checks.push({cohort, threshold, count, observation_rows: result.rows});
      }
    }
    const interactions = await evaluate(`(() => {
      const available = window.PAIRED_CENSUS_DATA.pairs[0];
      if (!available) return {empty:true, caseRendered:true, expectedIds:0, displayedIds:0};
      const cohort = document.getElementById('cohort'); cohort.value = available.cohort; cohort.dispatchEvent(new Event('change'));
      const search = document.getElementById('case-search'); search.value = available.id; search.dispatchEvent(new Event('input'));
      const caseRendered = document.querySelectorAll('#case-detail article').length === 2;
      const button = document.querySelector('#mechanisms button[data-method="candidate"]:not(:disabled)');
      if (!button) throw new Error("Nonempty group has no clickable count");
      button.click();
      return {caseRendered, expectedIds: Number(button.textContent), displayedIds: document.querySelectorAll('#selected-ids button').length};
    })()`);
    assert(interactions.caseRendered); assert.equal(interactions.expectedIds, interactions.displayedIds);
    await evaluate(`document.getElementById('cohort').value='main'; document.getElementById('threshold').value='0.006'; document.getElementById('cohort').dispatchEvent(new Event('change')); window.scrollTo(0,0);`);
    async function screenshot(name) { const image = await call('Page.captureScreenshot', {format: 'png'}); await fs.writeFile(path.join(destination, name), Buffer.from(image.data, 'base64')); }
    await screenshot('desktop.png');
    const assets = await evaluate(`(async () => {
      const images = [...document.querySelectorAll('img')];
      for (const image of images) { image.loading = 'eager'; if (!image.src.startsWith('data:image/')) throw new Error('External image retained'); }
      await Promise.all(images.map(image => image.decode()));
      if(document.querySelectorAll('script[src]').length) throw new Error('External script retained');
      const links=[...document.querySelectorAll('a[href]')];
      if(links.some(link=>!link.href.startsWith('data:'))) throw new Error('External link retained');
      return images.map((image,index)=>({url:'embedded_chart_'+index,status:image.naturalWidth>0?200:0,width:image.naturalWidth,height:image.naturalHeight}));
    })()`);
    assert.equal(assets.length, 4); assert(assets.every(asset=>asset.status===200));
    assert.equal(externalRequests.length, 0, JSON.stringify(externalRequests));
    await call('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 1, mobile: true});
    await evaluate('window.scrollTo(0,0)');
    const mobile = await evaluate('({viewport:innerWidth,document:document.documentElement.scrollWidth})');
    assert(mobile.document <= mobile.viewport, JSON.stringify(mobile));
    await screenshot('mobile.png');
    assert.equal(errors.length, 0, JSON.stringify(errors));
    const report = {url, checked_at_utc: new Date().toISOString(), method_identity: identity, cohort_threshold_checks: checks,
      interactions, resource_policy: 'HTTP and HTTPS blocked; inline scripts/data/images and download links only', external_requests: externalRequests, assets_checked: assets.length, failed_assets: assets.filter(asset => asset.status !== 200), mobile, javascript_errors: errors};
    await fs.writeFile(path.join(destination, 'browser_check.json'), JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report));
  } finally { if (socket) socket.close(); browser.kill('SIGTERM'); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
