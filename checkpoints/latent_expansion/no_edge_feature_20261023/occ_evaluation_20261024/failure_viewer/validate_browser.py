"""Exercise the delivered standalone file in Chromium, with networking blocked."""
from __future__ import annotations

import json
from pathlib import Path
import time

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
OUT = HERE / 'validation'
CHROMIUM = '/root/.cache/ms-playwright/chromium-1155/chrome-linux/chrome'


def main():
    OUT.mkdir(exist_ok=True)
    errors, network, checks = [], [], []
    start = time.monotonic()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROMIUM, headless=True,
            args=['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        page = browser.new_page(viewport={'width': 1680, 'height': 1120}, device_scale_factor=1)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('console', lambda message: errors.append(message.text) if message.type == 'error' else None)

        def block_request(route):
            network.append(route.request.url)
            route.abort()

        page.route('http://**/*', block_request)
        page.route('https://**/*', block_request)
        page.goto((HERE / 'failure_viewer.html').as_uri(), wait_until='load', timeout=90000)
        page.wait_for_function('window.viewer && !window.viewer.state.loading', timeout=90000)
        page.evaluate('window.viewer.ready')
        loaded_seconds = time.monotonic() - start
        first = page.evaluate('({id:viewer.state.id,source:viewer.state.source,face:viewer.state.face,error:viewer.state.error,n:MANIFEST.cases.length,traceCount:document.getElementById("geometry").data.length})')
        assert first['n'] == 316 and first['source'] == 'fitted' and first['error'] is None and first['traceCount'] > 2, first
        checks.append({'check': 'offline_initial_load', 'seconds': loaded_seconds, 'state': first})
        page.screenshot(path=str(OUT / 'overview.png'), full_page=True)

        ids = page.evaluate('MANIFEST.cases.map(c=>c.id)')
        decoded = []
        for start_index in range(0, len(ids), 16):
            batch = page.evaluate('''async ids => {
                const output=[];
                for(const id of ids){
                  const bytes=Uint8Array.from(atob(CASE_DATA[id]),c=>c.charCodeAt(0));
                  const c=JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());
                  if(c.id!==id||!c.views.fitted||c.replay.matches_history!==true)throw new Error('Invalid '+id);
                  output.push({id:c.id,faces:c.views.fitted.faces.length,edges:c.views.fitted.edges.length});
                }
                return output;
            }''', ids[start_index:start_index + 16])
            decoded.extend(batch)
        assert len(decoded) == 316
        checks.append({'check': 'all_embedded_cases_decode_in_browser', 'count': len(decoded),
                       'faces': sum(x['faces'] for x in decoded), 'edges': sum(x['edges'] for x in decoded)})
        print('Browser decoded 316/316 cases', flush=True)

        page.locator('#cohort-filter').select_option('regressed')
        page.wait_for_function('viewer.filtered.length===29 && !viewer.state.loading')
        assert page.locator('#case-list button').count() == 29
        page.locator('#cohort-filter').select_option('all')
        page.locator('#stage-filter').select_option('solid_construction')
        page.wait_for_function('viewer.filtered.length===6 && !viewer.state.loading')
        assert page.locator('#case-list button').count() == 6
        assert page.locator('#attempt-table tr').count() > 0
        checks.append({'check': 'cohort_and_stage_filters', 'regressed': 29, 'solid': 6,
                       'solid_log': page.locator('#face-log-summary').inner_text()})
        page.locator('#stage-filter').select_option('all')
        page.locator('#search').fill('00024146')
        page.wait_for_function('viewer.state.id==="00024146"&&!viewer.state.loading')
        assert page.locator('#case-list button').count() == 1
        page.locator('#search').fill('')
        page.evaluate('viewer.setSelection(6,null,true)')
        assert '丢' in page.locator('#attempt-table').inner_text()
        checks.append({'check': 'known_lost_wire_case', 'id': '00024146',
                       'attempt_count': page.locator('#attempt-table tr').count(),
                       'mean_text': page.locator('#selected-mean').inner_text()})
        page.locator('#isolate').check()
        page.locator('#focus-face').click()
        page.wait_for_timeout(600)
        page.screenshot(path=str(OUT / 'lost_wire_face.png'), full_page=True)
        page.locator('#focus-gap').click()
        page.wait_for_timeout(500)
        assert page.evaluate('viewer.state.focus') == 'gap'
        page.screenshot(path=str(OUT / 'local_gap.png'), full_page=True)
        page.locator('#focus-all').click()
        page.locator('#geometry').scroll_into_view_if_needed()
        page.wait_for_timeout(700)
        box = page.locator('#geometry').bounding_box()
        assert box
        before = page.evaluate('JSON.stringify(document.getElementById("geometry")._fullLayout.scene._scene.getCamera())')
        x, y = box['x'] + box['width'] * .5, box['y'] + box['height'] * .5
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x + 110, y + 50, steps=12)
        page.mouse.up()
        page.wait_for_timeout(500)
        after = page.evaluate('JSON.stringify(document.getElementById("geometry")._fullLayout.scene._scene.getCamera())')
        assert after != before, 'Dragging did not change the 3D camera'
        checks.append({'check': 'camera_drag_and_gap_zoom', 'passed': True})

        page.locator('#source').select_option('raw')
        page.wait_for_function('document.getElementById("view-caption").textContent.includes("原始预测")')
        assert '0</strong> 已测 / <strong>0' not in page.locator('#metric-coverage').inner_html()
        page.locator('#source').select_option('gt')
        page.wait_for_function('document.getElementById("view-caption").textContent.includes("GT")')
        page.locator('#source').select_option('fitted')
        page.wait_for_function('document.getElementById("view-caption").textContent.includes("拟合后")')
        if page.locator('#show-gt').count():
            page.locator('#show-gt').check()
            page.wait_for_timeout(400)
        checks.append({'check': 'raw_gt_fitted_source_switch', 'passed': True})

        options = page.locator('#distribution-metric option').evaluate_all('(els)=>els.map(e=>e.value)')
        for metric in options:
            page.locator('#distribution-metric').select_option(metric)
            page.evaluate('viewer.renderDistribution()')
            assert page.locator('#distribution-stats').inner_text()
        page.locator('#distribution-metric').select_option('raw_mean')
        page.evaluate('viewer.renderDistribution()')
        assert page.evaluate('document.getElementById("distribution").data.length') == 2
        page.locator('#distribution-hist').click()
        page.evaluate('viewer.renderDistribution()')
        assert page.evaluate('document.getElementById("distribution").data[0].type') == 'histogram'
        page.locator('#distribution-log').check()
        page.evaluate('viewer.renderDistribution()')
        page.locator('#distribution-cdf').click()
        page.evaluate('viewer.renderDistribution()')
        checks.append({'check': 'all_distribution_metrics_hist_cdf_log', 'metrics': options})
        page.locator('#distribution-log').uncheck()
        page.evaluate('viewer.renderDistribution()')
        page.locator('#distribution').screenshot(path=str(OUT / 'distribution.png'))

        tested = []
        for sid in ['00003999', '00208610', '00356252', next(c['id'] for c in page.evaluate('MANIFEST.cases') if c['stage'] == 'roundtrip_boundary')]:
            page.evaluate('(id)=>viewer.loadCase(id)', sid)
            assert page.evaluate('viewer.state.error') is None
            tested.append(sid)
        checks.append({'check': 'representative_stages_and_missing_edge_cases', 'ids': tested})
        assert page.evaluate('viewer.cacheSize') <= 3
        page.locator('#next').click()
        page.wait_for_function('!viewer.state.loading')
        page.locator('body').click(position={'x': 5, 'y': 5})
        previous = page.evaluate('viewer.state.id')
        page.keyboard.press('ArrowRight')
        page.wait_for_function('!viewer.state.loading')
        assert previous != page.evaluate('viewer.state.id')
        checks.append({'check': 'navigation_buttons_and_arrow_key', 'passed': True})

        page.set_viewport_size({'width': 1000, 'height': 850})
        page.wait_for_timeout(400)
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 2')
        checks.append({'check': 'compact_viewport_1000px', 'passed': True})
        assert not network, network
        assert not errors, errors
        report = {'passed': True, 'html': str(HERE / 'failure_viewer.html'),
                  'elapsed_seconds': time.monotonic() - start, 'checks': checks,
                  'javascript_errors': errors, 'external_requests': network,
                  'browser': browser.version, 'all_cases_decoded': len(decoded)}
        (OUT / 'browser.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'passed': True, 'checks': len(checks), 'seconds': report['elapsed_seconds']}), flush=True)
        browser.close()


if __name__ == '__main__':
    main()
