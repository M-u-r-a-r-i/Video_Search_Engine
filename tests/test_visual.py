import numpy as np

from backend.visual import clean_ocr_lines, select_keyframes


def _frame(value):
    return np.full((27, 48), value, dtype=np.float32)


def test_keyframes_only_when_the_picture_changes():
    sigs = [_frame(10), _frame(11), _frame(12), _frame(200), _frame(201), _frame(50)]
    assert select_keyframes(sigs, interval=2, threshold=10, max_gap=60) == [0, 3, 5]


def test_keyframe_kept_periodically_on_a_static_screen():
    sigs = [_frame(10)] * 100                       # 200 s of the same slide
    keep = select_keyframes(sigs, interval=2, threshold=10, max_gap=60)
    assert keep == [0, 30, 60, 90]


def test_ocr_noise_from_photos_is_dropped():
    lines = ["Brown sugar", "日", "x", "HIGH FRUCTOSE CORN SYRUP (42%)", "blurry"]
    scores = [0.98, 0.9, 0.95, 0.91, 0.3]
    assert clean_ocr_lines(lines, scores) == "Brown sugar HIGH FRUCTOSE CORN SYRUP (42%)"
