import numpy as np
import pytest

from meatpick.config import load_config, make_rng
from meatpick.eval.tracking_eval import TrackingScorer
from meatpick.perception.tracker import Tracker, _angle_median, track_position_at
from meatpick.types import Detection, Frame

STEP = 200  # encoder counts per frame = 20 mm of belt travel
DT = 0.08  # seconds per frame at 250 mm/s


@pytest.fixture
def cfg():
    return load_config()


def mask(border=False):
    m = np.zeros((720, 1280), bool)
    m[300:340, 600:640] = True
    if border:
        m[-1, 600:640] = True  # touches the bottom edge: the piece is cut off by the frame
    return m


def det(x, y, angle=30.0, fat=1, border=False, touching=False, conf=0.9):
    return Detection(
        mask=mask(border), x_mm=x, y_mm=y, angle_deg=angle, fat_side=fat, length_mm=140.0,
        width_mm=85.0, est_weight_g=300.0, touching=touching, confidence=conf,
    )  # fmt: skip


def frame(k):
    return Frame(np.zeros((1, 1, 3), np.uint8), 10_000 + k * STEP, k * DT)


def feed(tracker, start_x, y, n, **kw):
    """One piece riding the belt: its detection moves 20 mm per frame."""
    out = []
    for k in range(n):
        out.append(tracker.update(frame(k), [det(start_x + 20 * k, y, **kw)]))
    return out


def test_one_piece_keeps_one_id_and_is_confirmed_on_its_second_hit(cfg):
    t = Tracker(cfg)
    seen = feed(t, 100, 300, 6)
    assert [len(s) for s in seen] == [1] * 6
    assert len({s[0].id for s in seen}) == 1
    assert seen[0][0].status == "tentative" and seen[1][0].status == "confirmed"


def test_two_pieces_side_by_side_keep_separate_ids(cfg):
    t = Tracker(cfg)
    ids = None
    for k in range(8):
        tracks = t.update(frame(k), [det(100 + 20 * k, 150), det(100 + 20 * k, 330)])
        cur = {round(tr.y_mm): tr.id for tr in tracks}
        ids = ids or cur
        assert cur == ids  # same two IDs on the same two pieces every frame


def test_assignment_is_by_proximity_not_by_list_order(cfg):
    t = Tracker(cfg)
    for k in range(3):
        t.update(frame(k), [det(100 + 20 * k, 150), det(100 + 20 * k, 330)])
    a, b = sorted(t.tracks, key=lambda tr: tr.y_mm)
    t.update(frame(3), [det(160, 331), det(160, 151)])  # reversed order
    after = {tr.id: tr for tr in t.tracks}
    assert abs(after[a.id].y_mm - 151) < 3 and abs(after[b.id].y_mm - 331) < 3


def test_a_detection_outside_the_gate_starts_a_new_track(cfg):
    t = Tracker(cfg)
    feed(t, 100, 300, 3)
    old = t.tracks[0].id
    t.update(frame(3), [det(160 + 60, 300)])  # 60 mm from where the track predicts
    assert {tr.id for tr in t.tracks} - {old}  # a new id appeared


def test_track_coasts_by_encoder_after_leaving_the_view_then_is_missed(cfg):
    t = Tracker(cfg)
    k = 0
    for k in range(10):  # piece rides to x ~ 280 and is then lost from view
        t.update(frame(k), [det(100 + 20 * k, 300)])
    tr = t.tracks[0]
    x_at_last, enc_at_last = tr.x_mm, tr.ref_encoder
    for j in range(k + 1, k + 40):  # no detections any more
        t.update(frame(j), [])
        if t.tracks:
            expect = x_at_last + (frame(j).encoder - enc_at_last) * 0.1
            assert t.tracks[0].x_mm == pytest.approx(expect, abs=1e-6)
    assert t.tracks == []  # past the end of the reach zone
    done = t.drain_finished()
    assert len(done) == 1 and done[0].status == "missed" and done[0].x_mm > 1000
    assert t.drain_finished() == []


def test_picked_tracks_are_removed_and_reported(cfg):
    t = Tracker(cfg)
    feed(t, 100, 300, 4)
    tid = t.tracks[0].id
    t.mark_assigned(tid)
    assert t.get(tid).status == "assigned"
    t.mark_picked(tid)
    assert t.tracks == [] and t.drain_finished()[0].status == "picked"


def test_only_confirmed_tracks_can_be_assigned(cfg):
    t = Tracker(cfg)
    t.update(frame(0), [det(100, 300)])
    t.mark_assigned(t.tracks[0].id)
    assert t.tracks[0].status == "tentative"


def test_a_one_frame_false_positive_never_becomes_a_track(cfg):
    t = Tracker(cfg)
    t.update(frame(0), [det(150, 300)])
    t.update(frame(1), [])
    assert t.tracks == []


def test_a_ghost_lost_in_the_interior_is_dropped_but_coasting_tracks_are_not(cfg):
    t = Tracker(cfg)
    for k in range(3):  # confirmed while mid-view (x ~ 120-160)
        t.update(frame(k), [det(100 + 20 * k, 300)])
    for j in range(3, 6):  # three frames with nothing detected, still in the interior
        t.update(frame(j), [])
    assert t.tracks == []  # dropped as a ghost
    t2 = Tracker(cfg)
    for k in range(10):
        t2.update(frame(k), [det(100 + 20 * k, 300)])
    for j in range(10, 20):  # piece has left the view; it must keep coasting
        t2.update(frame(j), [])
    assert len(t2.tracks) == 1


def test_a_cut_off_detection_does_not_overwrite_the_pose(cfg):
    t = Tracker(cfg)
    for k in range(3):
        t.update(frame(k), [det(100 + 20 * k, 300, angle=30.0)])
    good = t.tracks[0]
    x_before, ang = good.x_mm, good.angle_deg
    # the piece is now cut by the frame edge: its centroid and angle are unreliable
    t.update(frame(3), [det(160 + 8, 310, angle=85.0, border=True)])
    after = t.tracks[0]
    assert after.angle_deg == pytest.approx(ang) and after.hits == 4
    assert after.x_mm == pytest.approx(x_before + 20.0, abs=1e-6)  # pure dead reckoning


def test_first_clean_view_replaces_a_pose_that_started_partial(cfg):
    t = Tracker(cfg)
    # a piece entering from the top: its visible-part centroid lags the true centre by ~10 mm
    t.update(frame(0), [det(10, 300, angle=70.0, border=True)])
    t.update(frame(1), [det(40, 305, angle=70.0, border=True)])
    t.update(frame(2), [det(62, 320, angle=25.0)])  # first clean view
    tr = t.tracks[0]
    assert tr.angle_deg == pytest.approx(25.0) and tr.y_mm == pytest.approx(320.0)


def test_angle_median_is_circular():
    assert _angle_median([179.0, 1.0, 179.0, 1.0, 179.0]) == pytest.approx(179.0)
    wrap = _angle_median([178.0, 179.0, 1.0, 2.0, 0.0])
    assert min(wrap, 180 - wrap) < 2.0  # near the 0/180 seam, not 90


def test_noisy_angle_is_smoothed_by_the_median(cfg):
    t = Tracker(cfg)
    for k, a in enumerate([30, 31, 80, 29, 30]):  # one wild frame
        t.update(frame(k), [det(100 + 20 * k, 300, angle=float(a))])
    assert t.tracks[0].angle_deg == pytest.approx(30.0, abs=1.5)


def test_a_fragment_next_to_a_tracked_piece_does_not_spawn_a_duplicate(cfg):
    t = Tracker(cfg)
    for k in range(3):
        t.update(frame(k), [det(100 + 20 * k, 300)])
    for k in range(3, 8):  # every frame also reports a small fragment 20 mm from the piece
        t.update(frame(k), [det(100 + 20 * k, 300), det(100 + 20 * k + 15, 315)])
    assert len(t.tracks) == 1


def test_ids_are_never_reused(cfg):
    t = Tracker(cfg)
    first = feed(t, 100, 300, 3)[0][0].id
    t.mark_picked(first)
    t.update(frame(3), [det(300, 300)])
    t.update(frame(4), [det(320, 300)])
    assert t.tracks[0].id != first


def test_deadline_follows_the_belt_speed(cfg):
    t = Tracker(cfg)
    feed(t, 100, 300, 5)  # 20 mm / 0.08 s = 250 mm/s
    tr = t.tracks[0]
    assert t.speed_mm_s == pytest.approx(250.0, rel=0.02)
    assert tr.deadline_s == pytest.approx(frame(4).t_s + (1000 - tr.x_mm) / 250.0, rel=0.02)


def test_deadline_is_infinite_when_the_belt_is_stopped(cfg):
    t = Tracker(cfg)
    t.update(Frame(np.zeros((1, 1, 3), np.uint8), 10_000, 0.0), [det(100, 300)])
    t.update(Frame(np.zeros((1, 1, 3), np.uint8), 10_000, 0.5), [det(100, 300)])  # no travel
    assert t.tracks[0].deadline_s == float("inf")


def test_position_at_is_pure_dead_reckoning(cfg):
    t = Tracker(cfg)
    feed(t, 100, 300, 3)
    tr = t.tracks[0]
    x, y = track_position_at(tr, tr.ref_encoder + 1000, 0.1)
    assert (x, y) == (pytest.approx(tr.x_mm + 100.0), tr.y_mm)


def test_scorer_counts_switches_and_duplicates():
    from meatpick.types import Track

    def tr(i, x):
        return Track(i, x, 300.0, 0, 0.0, 1, 1.0, 0.0, "confirmed")

    s = TrackingScorer()
    piece = [{"id": 7, "x_mm": 100.0, "y_mm": 300.0, "visibility": 1.0}]
    s.frame(0, piece, [tr(1, 101.0)])
    s.frame(0, piece, [tr(2, 102.0)])  # another id takes over: one switch
    s.frame(0, piece, [tr(2, 100.0), tr(3, 110.0)])  # two tracks on one piece: duplicate
    out = s.summary()
    assert out["id_switches"] == 1 and out["pieces_with_duplicate_tracks"] == 1


def test_fresh_rendered_sequence_has_no_switches_or_duplicates():
    """End to end with the classical detector on newly rendered dense scenes (dev seeds)."""
    from meatpick.perception.calibration import Calibration
    from meatpick.perception.factory import Perceiver
    from meatpick.sim.calibration_target import render_calibration_set
    from meatpick.sim.camera import Camera
    from meatpick.sim.dataset_export import DEV_SEED_BASE, _scene_frames
    from meatpick.sim.dataset_export import test_cfg as dense_line_cfg  # alias: not a test

    cfg = load_config()
    ncfg = dense_line_cfg(cfg)
    rng = make_rng(31)
    cam = Camera(ncfg, rng)
    tilted, flat, _ = render_calibration_set(cam, ncfg, rng)
    perceiver = Perceiver(cfg, Calibration.calibrate(tilted, flat), prefer="classical")
    scorer = TrackingScorer()
    tracker = Tracker(cfg)
    for fr, truth in _scene_frames(ncfg, DEV_SEED_BASE + 60, 40, 1, True):
        tracks = tracker.update(fr, perceiver.perceive(fr))
        pieces = [{"id": p.id, "x_mm": p.x_mm, "y_mm": p.y_mm, "visibility": p.visibility}
                  for p in truth.pieces]  # fmt: skip
        scorer.frame(0, pieces, tracks)
    out = scorer.summary()
    assert out["pieces"] >= 3 and out["pieces_with_duplicate_tracks"] == 0
    assert out["id_switches"] <= 1


def _box_detection(cfg, x_center, half_len=80.0, y=300.0, span=None):
    """Detection of a piece lying along belt x: what a detector sees when the frame cuts it."""
    top, fov = span or (
        0.0,
        cfg["camera"]["height_px"] / (cfg["camera"]["width_px"] / cfg["camera"]["fov_mm"][0]),
    )
    lo, hi = max(x_center - half_len, top), min(x_center + half_len, fov)
    if hi <= lo:
        return None
    m = np.zeros((720, 1280), bool)
    m[300:340, 600:640] = True
    if hi >= fov:
        m[-3, 600:640] = True  # cut by the bottom edge (masks stop a few pixels short of it)
    if lo <= top:
        m[2, 600:640] = True  # cut by the top edge
    d = det(0.5 * (lo + hi), y, angle=0.0)
    d.mask = m
    d.length_mm, d.width_mm = hi - lo, 85.0
    return d


def test_a_piece_leaving_the_view_creates_no_ghost_and_keeps_its_true_position(cfg):
    """Its visible centroid slides up to half the hidden length behind the true centre."""
    t = Tracker(cfg)
    ghosts = 0
    for k in range(60):
        xc = -60.0 + 20 * k  # true centre: enters at the top, exits at the bottom
        d = _box_detection(cfg, xc)
        t.update(frame(k), [d] if d else [])
        if k > 8:
            ghosts = max(ghosts, len(t.tracks) - 1)
    assert ghosts == 0
    done = t.drain_finished()
    # the one track kept coasting on the encoder and left the reach zone as a miss at its true x
    assert len(done) == 1 and done[0].status == "missed"
    assert done[0].x_mm == pytest.approx(-60.0 + 20 * 54, abs=2.0)  # retired at x ~ 1020


def test_a_piece_entering_at_the_top_is_tracked_from_its_partial_views(cfg):
    t = Tracker(cfg)
    for k in range(8):
        d = _box_detection(cfg, -50.0 + 20 * k)
        t.update(frame(k), [d] if d else [])
    assert len(t.tracks) == 1 and t.tracks[0].status == "confirmed"
    assert t.tracks[0].x_mm == pytest.approx(-50.0 + 20 * 7, abs=3.0)  # clean views fixed the pose


def test_a_side_clipped_piece_is_not_dragged_back_as_it_slides_out_of_the_bottom(cfg):
    """A piece near the belt edge is cut at the side for its whole transit, so it never gets a
    clean view; its visible centroid lags the true centre once the bottom edge starts cutting it."""
    t = Tracker(cfg)
    for k in range(25):
        d = _box_detection(cfg, -20.0 + 20 * k)
        if d is not None:
            d.mask[300:340, 0] = True  # side clipped on every frame
        t.update(frame(k), [d] if d else [])
    x_true = -20.0 + 20 * 24
    tr = t.tracks[0]
    assert tr.x_mm + (frame(24).encoder - tr.ref_encoder) * 0.1 == pytest.approx(x_true, abs=8.0)


def test_nothing_is_spawned_from_a_detection_cut_by_the_exit_edge(cfg):
    t = Tracker(cfg)
    d = _box_detection(cfg, 330.0)  # half out of the bottom edge, never seen before
    t.update(frame(0), [d])
    t.update(frame(1), [_box_detection(cfg, 350.0)])
    assert t.tracks == []


class _SpanCalibration:
    """Stands in for a calibration of a tilted camera: the view spans belt x 17.5 to 358 mm."""

    def view_span_x_mm(self):
        return 17.5, 358.0


def test_tracker_takes_its_view_span_from_the_calibration(cfg):
    """A tilted camera sees belt x 17.5-358, not the nominal 0-337.5."""
    t = Tracker(cfg, _SpanCalibration())
    assert (t.x_top, t.x_bot) == (17.5, 358.0)
    nominal = Tracker(cfg)
    assert nominal.x_top == 0.0 and nominal.x_bot == pytest.approx(337.5)


def test_no_ghost_while_a_piece_slides_out_of_a_tilted_view(cfg):
    span = (17.5, 358.0)
    t = Tracker(cfg, _SpanCalibration())
    for k in range(60):
        d = _box_detection(cfg, -60.0 + 20 * k, span=span)
        t.update(frame(k), [d] if d else [])
        if k > 8:
            assert len(t.tracks) <= 1
    done = t.drain_finished()
    assert len(done) == 1 and done[0].x_mm == pytest.approx(-60.0 + 20 * 54, abs=2.0)
