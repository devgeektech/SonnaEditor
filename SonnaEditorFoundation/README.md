# Sonna Editor Foundation Model

Repo-local folder for the active Sonna Editor foundation checkpoint.
Do not store RAW photos or generated training datasets here.

Files:
- `foundation_manifest.json`: points to the active checkpoint.
- `checkpoints/*.ckpt`: versioned foundation checkpoints.
- `checkpoints/*.json`: matching checkpoint sidecars.

Current active checkpoint in this branch:
`checkpoints/foundation-sonna-raw-xmp-004-visual.ckpt`.

Checkpoint binaries are tracked by the parent SonnaEditor repo through Git LFS.
