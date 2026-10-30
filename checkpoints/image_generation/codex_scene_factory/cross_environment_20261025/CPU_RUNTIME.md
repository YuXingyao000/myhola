# Authorized rendering resource change

The user selected CPU rendering because all eight GPUs are occupied by training.
All subsequent Blender previews and final renders must use `--device CPU` and
`--threads 8`. GPU visibility is disabled in the process environment; do not
override it or invoke CUDA/OptiX rendering. Keep denoising on CPU
(`cycles.denoiser = 'OPENIMAGEDENOISE'`, `cycles.denoising_use_gpu = False`),
and use `render.compositor_device = 'CPU'` if scene compositing is enabled.

This is an execution-resource change only. Keep the scene brief, six review
views, seed 8, quick-preview sampling and formal 128-sample rendering protocol.
Do not reduce image resolution or sampling merely to compensate for CPU speed.

The prior repair turn was interrupted by the host to change resources. Its
edits remain on disk. Continue the same repair and original review findings;
do not restart scene creation, revert existing work or count this interruption
as a new quality-repair iteration. The host will independently rebuild and
render all six review views after this repair completes.
