# NOTES

Assumption changes and choices made while building (see PRD working rule 9).

- `ultralytics` is an optional extra (`pip install -e .[yolo]`) so the base install stays light; classical detector works without it.
- `ffmpeg` was not on PATH at scaffold time; needed for #18 (`brew install ffmpeg`).
