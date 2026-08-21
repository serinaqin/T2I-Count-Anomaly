import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, maximum_filter


def featuremap_saliency(act, cond_index=1, normalize=True):
    """Per-pixel saliency of an attention feature map. (B,T,C)->cond half
    (T,C)-> L2 norm over channels -> (H,W) with H=W=round(sqrt(T)). If
    normalize, min-max to [0,1] (for eyeball/peak-count); if not, return the
    raw per-pixel energy (preserves 'amount', needed for magnitude probing)."""
    a = act.detach().cpu().float().numpy() if hasattr(act, "detach") \
        else np.asarray(act, float)
    if a.ndim == 3:
        a = a[cond_index] if a.shape[0] > cond_index else a[-1]  # (T, C)
    norm = np.linalg.norm(a, axis=-1)                            # (T,)
    h = int(round(np.sqrt(norm.shape[0])))
    sal = norm[: h * h].reshape(h, h)
    if not normalize:
        return sal
    lo, hi = sal.min(), sal.max()
    return (sal - lo) / (hi - lo) if hi > lo else np.zeros_like(sal)


def saliency_to_instance_count(sal, thresh=0.5, min_size=1) -> int:
    """Count 4-connected blobs (>= min_size pixels) in the thresholded
    saliency map = an internal estimate of how many object instances the
    feature map represents."""
    mask = np.asarray(sal) >= thresh
    labeled, n = ndimage.label(mask)
    if n == 0:
        return 0
    sizes = ndimage.sum(mask, labeled, index=range(1, n + 1))
    return int((sizes >= min_size).sum())


def find_peaks(sal, sigma=1.0, min_distance=2, thresh_rel=0.5):
    """Local maxima of a Gaussian-smoothed saliency map, kept if above
    thresh_rel * max. Counts distinct object centers even when their blobs
    merge (two adjacent cats = two peaks in one bright region) and ignores
    texture specks. Returns an array of (row, col) peak centers."""
    s = gaussian_filter(np.asarray(sal, float), sigma)
    if s.max() <= 0:
        return np.empty((0, 2), int)
    mx = maximum_filter(s, size=2 * min_distance + 1, mode="nearest")
    peaks = (s == mx) & (s >= thresh_rel * s.max())
    labeled, n = ndimage.label(peaks)
    if n == 0:
        return np.empty((0, 2), int)
    centers = ndimage.center_of_mass(peaks, labeled, range(1, n + 1))
    return np.array([[int(round(r)), int(round(c))] for r, c in centers])


def count_peaks(sal, sigma=1.0, min_distance=2, thresh_rel=0.5) -> int:
    """Number of object-center peaks in the saliency map (calibrated instance
    count; robust to merged objects and texture, unlike blob counting)."""
    return len(find_peaks(sal, sigma, min_distance, thresh_rel))


def grid_pool_2d(m, g):
    """Average-pool a 2D map into a g x g grid, flattened to (g*g,). g=1 gives
    the global mean (spatially blind); larger g preserves spatial layout. Used
    to test whether a signal (e.g. object count) lives in spatial structure."""
    m = np.asarray(m, float)
    H, W = m.shape
    ys = np.linspace(0, H, g + 1).astype(int)
    xs = np.linspace(0, W, g + 1).astype(int)
    out = np.zeros((g, g))
    for i in range(g):
        for j in range(g):
            block = m[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            out[i, j] = block.mean() if block.size else 0.0
    return out.ravel()
