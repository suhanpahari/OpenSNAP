import numpy as np
from PIL import Image


def look_at(eye, target, up=(0, 0, 1)):
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, (0, 1, 0))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, -u, f])


def splat(xyz, rgb, R, eye, size, focal, radius):
    cam = (xyz - eye) @ R.T
    z = cam[:, 2]
    ok = z > 0.05
    cam, rgb, z = cam[ok], rgb[ok], z[ok]
    u = (focal * cam[:, 0] / z + size / 2).astype(np.int32)
    v = (focal * cam[:, 1] / z + size / 2).astype(np.int32)
    img = np.zeros((size, size, 3), np.float32)
    depth = np.full((size, size), np.inf, np.float32)
    order = np.argsort(-z)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            uu, vv = u[order] + dx, v[order] + dy
            m = (uu >= 0) & (uu < size) & (vv >= 0) & (vv < size)
            img[vv[m], uu[m]] = rgb[order][m]
            depth[vv[m], uu[m]] = np.minimum(depth[vv[m], uu[m]], z[order][m])
    return img, np.isfinite(depth)


def outline(hit, width=3):
    grown = hit.copy()
    for _ in range(width):
        g = grown.copy()
        g[1:] |= grown[:-1]; g[:-1] |= grown[1:]; g[:, 1:] |= grown[:, :-1]; g[:, :-1] |= grown[:, 1:]
        grown = g
    return grown & ~hit


def render_object(coord, color, mask, n_views=3, size=448, context=0.8, fade=0.7, elev=35.0, spacing=0.02,
                  style="highlight", zoom=3.2):
    """Object-centric multi-view renders. style='highlight': real-color context + red outline; 'isolate': faded context."""
    if style == "highlight":
        context, fade = 2.5, 0.0
    color = np.full_like(coord, 180.0) if color is None else color
    obj = coord[mask]
    center = (obj.min(0) + obj.max(0)) / 2
    radius = max(np.linalg.norm(obj.max(0) - obj.min(0)) / 2, 0.15)
    near = np.linalg.norm(coord - center, axis=1) < radius * (1 + context)
    ctx = near & ~mask
    dist = radius * (zoom if style == "highlight" else 2.6)
    views = []
    base = np.degrees(np.arctan2(*(coord.mean(0)[:2] - center[:2])[::-1]))
    spread = 35.0 if style == "highlight" else 360.0 / n_views
    for k in range(n_views):
        az = np.radians(base + spread * (k - (n_views - 1) / 2))
        el = np.radians(elev)
        eye = center + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        R = look_at(eye, center)
        focal = size / 2 / np.tan(np.radians(30))
        rad = int(np.clip(np.ceil(focal * spacing / dist), 1, 6))
        c_ctx = ctx & (((coord - eye) @ R[2]) > ((obj - eye) @ R[2]).min() - 0.05)
        bg, bg_hit = splat(coord[c_ctx], color[c_ctx] * (1 - fade) + 255 * fade, R, eye, size, focal, rad)
        fg, fg_hit = splat(obj, color[mask], R, eye, size, focal, rad)
        img = np.full((size, size, 3), 255.0, np.float32)
        img[bg_hit] = bg[bg_hit]
        img[fg_hit] = fg[fg_hit]
        if style == "highlight":
            img[outline(fg_hit)] = (255, 0, 0)
        views.append(Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)))
    return views
