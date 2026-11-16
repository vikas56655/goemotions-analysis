import numpy as np
from src.visualize import (save_label_frequency, save_cooccurrence,
                           save_f1_per_label, save_threshold_curve)


def test_visualizations_write_real_images(tmp_path):
    y = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]])
    p = np.array([[0.8, 0.1, 0.7], [0.2, 0.9, 0.2], [0.6, 0.3, 0.1]])
    names = ["joy", "sadness", "neutral"]
    save_label_frequency(y, names, tmp_path / "frequency.png")
    save_cooccurrence(y, names, tmp_path / "cooccurrence.png")
    save_f1_per_label(y, p, 0.5, names, tmp_path / "f1.png")
    save_threshold_curve([{"threshold": 0.3, "micro_f1": 0.7},
                          {"threshold": 0.5, "micro_f1": 0.9}],
                         0.5, tmp_path / "threshold.png")
    for path in tmp_path.glob("*.png"):
        assert path.stat().st_size > 1000
    assert len(list(tmp_path.glob("*.png"))) == 4
