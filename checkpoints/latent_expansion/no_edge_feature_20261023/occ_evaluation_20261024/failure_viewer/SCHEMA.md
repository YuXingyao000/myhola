# Browser payload v1

`window.MANIFEST` = `{version, title, summary, cases:[{id,stage,cohort,failed_faces:[int], reasons:[string], raw_mean,raw_p95,raw_max,endpoint_max,replay_match,replay_mean,replay_p95,replay_max}], distributions:{raw_mean:{valid:[number],failed:[number],unknown_valid:int,unknown_failed:int}, raw_p95:..., raw_max:..., endpoint_max:..., replay_mean:{failed:[],valid:[],unknown_failed:int,unknown_valid:int}, replay_max:...}, definitions:{...}}`.

`window.CASE_DATA` maps id to gzip-compressed UTF8 JSON base64. Decode with DecompressionStream. Decompressed case:

```
{
 id, stage, cohort,
 diagnostics: {paths: [original path dicts, with faces/attempts]},
 replay: {status, matches_history: bool, mismatch_fields:[], note, error?},
 views: {
   fitted: {label, faces:[grid[N][M][3]], edges:[points[K][3]], connectivity:[[edge,face1,face2]], owner_records:[...], endpoint_records:[...]},
   raw: {...},
   gt: {...},
   baseline: {...}
 },
 metrics: {raw_mean,raw_p95,raw_max,endpoint_max,replay_mean,replay_p95,replay_max}
}
```

Views are independently optional; fitted may be missing after replay error. Face IDs agree across views. Edge IDs are view-specific. For fitted they match historical `input_edge_ids`; for raw/GT/baseline use connectivity to find edges owned by selected face. Do not index raw edges with post IDs. Current artifact includes fitted/raw/gt; baseline is optional and omitted to limit file size.

Additional metrics/distributions: `endpoint_mean`, `replay_endpoint_mean`, `replay_endpoint_max`. Replay distributions have `valid:[], unknown_valid:3179, not_measured_valid:3179`: successful objects were not replayed, not zero-distance. Any unknown projection causes replay CAD owner aggregates to be null; relation-level known measurements remain visible.

Endpoint clarification after inspecting diagnostic replay (no production rule change): fitted `endpoint_records` now chooses the smaller of the original nearest-other-edge proxy and the curve's own opposite endpoint (self seam). Each record adds `own_seam_distance`, `original_neighbor_distance`, `original_neighbor_edge`, `original_neighbor_endpoint`, and `basis` identifies the self candidate. The original records are preserved as `endpoint_other_edge_records`. `replay_endpoint_mean/max` use the new inclusive candidate, while `replay_endpoint_other_mean/max` preserve original proxy metrics, `replay_endpoint_other_unknown` counts its null records. UI must call the default “最近连接候选（含自身首尾）”, a lower bound, not proven wire closure. Show own seam and original other-neighbor distances for the selected edge. Raw records remain the historical GT-assisted protocol. Raw `endpoint_unknown` counts only null endpoint records; `endpoint_empty_owner_faces` separately counts faces with no incident input edges (different units).

owner_records: `{edge,owner,mean,p95,max, distances:[number|null], closest:[[x,y,z]|null], valid_points,unknown_points}`. `closest` and `distances` optional on non-fitted views. A connector joins `view.edges[edge][point_index]` to `closest[point_index]`.

endpoint_records: `{owner,edge,endpoint:0|-1,neighbor_edge:int|null,neighbor_endpoint:0|-1|null,distance:number|null,basis:string}`. `neighbor_*` optional raw; no fabricated neighbor. Endpoints and distances are not GT differences.

Stage codes: `trimming`, `solid_construction`, `roundtrip_boundary`. Cohort `regressed` (32 valid →64 fail), `both_failed` (both fail). Native-valid controls remain in the distribution background only.

Raw p95 = max edge-owner P95, fitted replay_p95 likewise. Raw/fitted mean = mean(edge-owner mean). Projection metrics use fitted bounded analytic surface with 64 parameter samples; they do not assert supremum or exact OCC tolerance. Endpoint metrics are local nearest-neighbor proxies, not wire closure proofs. Cases may lack finite metric; null must remain unknown. Definitions in SPEC.md take precedence.

HTML template placeholders: `__PLOTLY_JS__`, `__FONT_STYLE__`, `__MANIFEST_JSON__`, `__CASE_DATA_JSON__`, `__APP_JS__`. template.html has `<script>window.MANIFEST=__MANIFEST_JSON__;window.CASE_DATA=__CASE_DATA_JSON__;</script>` and separate app placeholder. Root packs asset; UI agent owns only template.html and app.js.
