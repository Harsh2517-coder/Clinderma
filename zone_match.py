"""Matching helpers shared by the accuracy evaluation and the spot-filter training."""
def touches(a, b):      # IoU >= 0.2, or the centre of one box lies inside the other
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    iou = i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)
    ca, cb = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2), ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    inside = lambda c, r: r[0] <= c[0] <= r[2] and r[1] <= c[1] <= r[3]
    return iou >= 0.2 or inside(ca, b) or inside(cb, a)


def zone_of(cx, cy, zm): return next((z for z, m in zm.items() if 0 <= int(cy) < m.shape[0] and 0 <= int(cx) < m.shape[1] and m[int(cy), int(cx)]), None)


