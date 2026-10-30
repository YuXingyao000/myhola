## Execution constraints for this batch

Other independent room tasks are running concurrently. Use only the GPU index
provided in your runtime; do not launch other Codex sessions or use other GPUs.

Every asset `fetch` must acquire the same shared cache lock, including fetches
for assets already cached. Substitute the actual runtime paths in this command:

```text
flock -x ASSET_CACHE/.fetch.lock PYTHON ASSETS_SCRIPT fetch ...
```

Here `ASSET_CACHE`, `PYTHON`, and `ASSETS_SCRIPT` mean `runtime.asset_cache`,
`runtime.python`, and `runtime.assets_script`. Keep the lock held until that
fetch command exits. Do not bypass the lock or remove it when another task is
using it. Asset searches and room build/render operations do not need this lock.
All actual dependencies must still be packaged into your own template directory.
