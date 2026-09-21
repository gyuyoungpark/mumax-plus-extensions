# Data archive status

This is a source-code update. Raw magnetization trajectories, GPU checkpoints,
build binaries, and ongoing validation outputs are not included. No persistent
complete data-archive identifier is assigned by this update.

Historical `src/sim*.py` scripts refer to archives under `data/`; the revision
drivers write separate output directories. Analysis scripts that require those
inputs are not data-independent tests. A successful source import does not
certify the old results, and the presence of an old summary is not proof that its
raw inputs are complete or match the current code.

The earlier blanket statement that every legacy script is self-contained and
checkpoint-resumable is withdrawn. Review its source and protect existing data
before running a historical script. For new validation, use the identity-checked
drivers and the prerequisites described in `SOURCE_UPDATE.md`.

Future data releases must identify the actual source, build, inputs, and completed
raw outputs together. A current source commit must not be assigned retrospectively
to old data without evidence of that connection.
