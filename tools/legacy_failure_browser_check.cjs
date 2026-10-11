/* Browser verification for the frozen offline failure atlas. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {pathToFileURL} = require('node:url');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE_PATH || 'playwright');

(async () => {
  const root = path.resolve(process.argv[2]);
  const out = path.join(root, 'validation');
  fs.mkdirSync(out, {recursive: true});
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'selected_cases.json')));
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || undefined,
    args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
  const page = await browser.newPage({viewport: {width: 1500, height: 1100}});
  const errors = [], requests = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('requestfailed', request => requests.push({url: request.url(), error: request.failure()}));
  const base = pathToFileURL(path.join(root, 'page/index.html')).href;
  await page.goto(base + '#00001926');
  await page.waitForFunction(() => window.LEGACY_VIEWER_READY === '00001926');
  assert.equal(await page.locator('canvas').count(), 3);
  assert.match(await page.locator('#viewer-status').innerText(), /三栏相机/);
  assert.equal(await page.locator('.case-item').count(), 50);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  const visited = [];
  for (const item of manifest.cases) {
    await page.evaluate(id => { location.hash = id; }, item.id);
    await page.waitForFunction(id => window.LEGACY_VIEWER_READY === id, item.id);
    assert.equal(await page.locator('#case-title').innerText(), item.id);
    assert.match(await page.locator('#observations').innerText(), /有效 solid 未通过/);
    const canvases = await page.locator('canvas').evaluateAll(nodes => nodes.map(node => ({
      width: node.width, height: node.height, webgl: Boolean(node.getContext('webgl2') || node.getContext('webgl'))
    })));
    assert(canvases.length === 3 && canvases.every(canvas => canvas.width > 0 && canvas.height > 0 && canvas.webgl));
    visited.push(item.id);
  }
  await page.selectOption('#filter', 'solid');
  assert.equal(await page.locator('.case-item').count(), 1);
  await page.locator('.case-item').click();
  await page.waitForFunction(() => window.LEGACY_VIEWER_READY === '00480996');
  assert.match(await page.locator('#observations').innerText(), /容差敏感/);
  await page.selectOption('#filter', 'topology_exact');
  assert.equal(await page.locator('.case-item').count(), 35);
  await page.selectOption('#filter', 'mesh_missing');
  assert.equal(await page.locator('.case-item').count(), 13);
  await page.selectOption('#filter', 'all');
  await page.fill('#search', '00001926');
  assert.equal(await page.locator('.case-item').count(), 1);
  await page.locator('.case-item').click();
  await page.waitForFunction(() => window.LEGACY_VIEWER_READY === '00001926');
  await page.fill('#search', '');
  const canvasImages = () => page.locator('canvas').evaluateAll(nodes => nodes.map(node => node.toDataURL()));
  const before = await canvasImages();
  const box = await page.locator('#canvas-raw').boundingBox();
  await page.mouse.move(box.x+box.width*.5, box.y+box.height*.5);
  await page.mouse.down(); await page.mouse.move(box.x+box.width*.7, box.y+box.height*.65, {steps:8}); await page.mouse.up();
  const after = await canvasImages();
  assert(before.every((value,index) => value !== after[index]), 'All three cameras must change on one-pane drag');
  await page.check('#wire-only');
  const wire = await canvasImages();
  assert(after.every((value,index) => value !== wire[index]), 'Wire toggle must affect every pane');
  await page.uncheck('#wire-only');
  await page.check('#xray'); await page.uncheck('#xray');
  await page.check('#face-colors'); await page.uncheck('#face-colors');
  await page.click('[data-camera=iso]');
  await page.fill('#notes', 'Browser verification note');
  assert.match(await page.locator('#notes-status').innerText(), /已保存/);
  await page.reload();
  await page.waitForFunction(() => window.LEGACY_VIEWER_READY === '00001926');
  assert.equal(await page.locator('#notes').inputValue(), 'Browser verification note');
  await page.fill('#notes', '');
  await page.screenshot({path:path.join(out,'desktop.png'), fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  await page.screenshot({path:path.join(out,'mobile.png'), fullPage:true});
  await page.setViewportSize({width:1500,height:1100});
  for (const id of ['00009683','00480996','00038614','00439331']) {
    await page.evaluate(value => {location.hash=value;}, id);
    await page.waitForFunction(value => window.LEGACY_VIEWER_READY === value, id);
    await page.evaluate(() => window.scrollTo(0,0));
    await page.screenshot({path:path.join(out,id+'.png'),fullPage:false});
  }
  assert.deepEqual(errors, []);
  assert.deepEqual(requests, []);
  const report = {browser:await browser.version(), case_count:visited.length, visited,
    javascript_errors:errors, failed_requests:requests,
    checks:['50 cases load offline with 3 WebGL panes','all cases preserve original invalid-solid label',
      'type, exact-pair and missing-mesh filters','ID search','one-pane drag updates all 3 cameras',
      'wire/xray/face-color toggles','notes persist across reload','desktop and 390px layout without horizontal overflow'],
    timestamp:new Date().toISOString()};
  fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({case_count:visited.length,javascript_errors:errors,failed_requests:requests,checks:report.checks.length}));
  await browser.close();
})().catch(error => {console.error(error);process.exit(1);});
