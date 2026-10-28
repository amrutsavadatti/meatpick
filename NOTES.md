# NOTES

Assumption changes and choices made while building (see PRD working rule 9).

- `ultralytics` is an optional extra (`pip install -e .[yolo]`) so the base install stays light; classical detector works without it.
- `ffmpeg` was not on PATH at scaffold time; needed for #18 (`brew install ffmpeg`).
- Dataset labels are one full-image box per photo (`0 0.5 0.5 1 1`, Beef/Pork), so it is an image classifier in YOLO format. Textures come from random crops, using a black-border mask instead of labels. Issue #16 (precision/recall vs dataset labels) is therefore degenerate; I will report it as "does the model fire on real meat photos" and say so in the report.
- Textures use `train` only and the Pork class only (`textures.classes: [1]`); `valid` and `test` stay untouched for #16. Dataset has 171 source photos x 3 augmented copies.
- Dataset "fat" patches are pale pink; they are tinted toward cream in `TextureBank.sample`. Real lean patches run magenta, so #3 needs a per-piece colour shift toward pork pink.
