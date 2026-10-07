# Bundled Lab build (placeholder)

This folder is where the built static files of `apps/lab` (index.html and
its assets) are copied before a release, so `pta.show(..., mode="local")`
and `pta.watch` can serve the full Lab from 127.0.0.1, offline.

While it holds only this README, the local server serves the package's
built-in raw viewer at `/` instead, and `mode="auto"` uses links.

To try a local Lab build without repackaging: `PTA_LAB_DIR=path/to/dist`.
