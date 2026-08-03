import numpy as np
from src.spatial import (featuremap_saliency, saliency_to_instance_count,
                         find_peaks, count_peaks)


def test_featuremap_saliency_shape_and_peak():
    # (B=2, T=4, C=3): cond half has one strong pixel at token 2
    cond = np.zeros((4, 3)); cond[2] = [3.0, 4.0, 0.0]   # norm 5
    act = np.stack([np.zeros((4, 3)), cond])
    sal = featuremap_saliency(act, cond_index=1)
    assert sal.shape == (2, 2)
    assert sal.max() == 1.0
    assert np.unravel_index(np.argmax(sal), sal.shape) == (1, 0)  # token 2


def test_saliency_to_instance_count_two_blobs():
    sal = np.zeros((5, 5))
    sal[0, 0] = 1.0                     # blob 1
    sal[4, 4] = 1.0; sal[4, 3] = 1.0    # blob 2
    assert saliency_to_instance_count(sal, thresh=0.5) == 2


def test_saliency_min_size_filters_specks():
    sal = np.zeros((5, 5)); sal[0, 0] = 1.0; sal[4, 4] = 1.0; sal[4, 3] = 1.0
    assert saliency_to_instance_count(sal, thresh=0.5, min_size=2) == 1


def test_count_peaks_two_separated():
    sal = np.zeros((20, 20)); sal[5, 5] = 1.0; sal[15, 15] = 1.0
    assert count_peaks(sal, sigma=1.0, min_distance=2, thresh_rel=0.3) == 2


def test_count_peaks_empty():
    assert count_peaks(np.zeros((10, 10))) == 0


def test_find_peaks_returns_centers():
    sal = np.zeros((20, 20)); sal[5, 5] = 1.0; sal[15, 15] = 1.0
    pk = find_peaks(sal, sigma=1.0, min_distance=2, thresh_rel=0.3)
    assert pk.shape == (2, 2)
    rows = sorted(pk[:, 0].tolist())
    assert rows == [5, 15]
