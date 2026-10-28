# NOTES

Assumption changes and choices made while building (see PRD working rule 9).

- `ultralytics` is an optional extra (`pip install -e .[yolo]`) so the base install stays light; classical detector works without it.
- `ffmpeg` was not on PATH at scaffold time; needed for #18 (`brew install ffmpeg`).
- Dataset labels are one full-image box per photo (`0 0.5 0.5 1 1`, Beef/Pork), so it is an image classifier in YOLO format. Textures come from random crops, using a black-border mask instead of labels. Issue #16 (precision/recall vs dataset labels) is therefore degenerate; I will report it as "does the model fire on real meat photos" and say so in the report.
- Textures use `train` only and the Pork class only (`textures.classes: [1]`); `valid` and `test` stay untouched for #16. Dataset has 171 source photos x 3 augmented copies.
- Dataset "fat" patches are pale pink; they are tinted toward cream in `TextureBank.sample`. Real lean patches run magenta, so #3 needs a per-piece colour shift toward pork pink.
- Pieces are generated with width <= 0.75 x length so the long axis (and therefore the grasp yaw) is well defined; the PRD ranges allow near-round pieces where no long axis exists. Edge wobble skips the k=2 harmonic because it tilts the principal axis.
- Piece weight is sampled in the PRD 200-400 g range and thickness derived from it (clipped to 15-30 mm), so weight and thickness stay consistent with the area.
- Camera/belt layout: the belt is 600 mm wide and the image is 1280x720, so the camera must see the full belt width across the image columns and ~337 mm of travel down the image rows (`camera.fov_mm` is `[across, along]`). Belt frame -> image is a 90 degree proper rotation (see `sim/geometry.py`); belt y runs right-to-left across the image.
- Fat side is defined relative to the normal of the reduced [0,180) angle. Converting between image and belt frames can wrap that angle and flip the sign; `geometry.fat_side_image_to_belt` is the one place that handles it, and perception (#9) must use the same rule via calibration.
- Cluster arrivals: burst_prob 0.1, size 2-3, spacing [-15, 30] mm. The PRD's 50 +/- 10 pieces per 60 s is about one standard deviation with heavier bursts, so bursts are kept mild by default.
- Camera model: scene is composited in the ideal frame on a canvas with a 100 px margin, then warped once (tilt homography + OpenCV radial distortion, `Camera.project` is the forward model and the remap maps are its inverse). Lens/geometry parameters are drawn once per Camera instance (a real camera is fixed); exposure and white balance drift per frame; noise is drawn per frame. Dataset export (#6) should create several Camera instances to get geometry variety.
- Sprites are drawn at whole pixels; truth polygons are built from the drawn position (sub-pixel outline from a 4x upsampled alpha) so truth matches pixels to <= 2 px (measured max 1.9 px, 0.7% area bias). The exact center stays in `x_mm, y_mm`.
- Render speed on this machine: ~25 ms/frame (~40 fps) at 1280x720, above the 15 fps target. The first frame interval after a Camera is created depends on when it started watching; later intervals are exactly 20 mm.
- `outputs/debug/camera_sample.mp4` is written with OpenCV's mp4v codec because ffmpeg is not installed here; the final demo video (#18) needs ffmpeg for H.264.
