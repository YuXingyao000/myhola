# Runtime Tool Configuration

```json
{runtime_json}
```

The caller supplies these paths. Run assets.py and preview.py with Python; see
--help for arguments. The cache reuses downloads; the template must contain the
assets it actually needs. Do not print authentication environment variables.
Keep tool source code, the fixed test panel, and source CAD files unchanged.
Network and command access depend on the current runtime permissions.

Quick preview command:

```text
{quick_preview_command}
```

When Blender and the STL directory are configured, use quick previews when they
help development; there is no minimum preview count. The host runs the final
review preset after building, then starts a separate image-review turn. Follow
the current stage instructions; do not run final review yourself.

When Blender and the STL directory are not configured, previews are unavailable.
Do not claim that they ran; identify them as unverified at delivery.
