"""
Gargantua — Schwarzschild ışık bükülmesi çekirdeği (saf matematik).

Birimler: rs = 1 (Schwarzschild yarıçapı). Her pikselin ışını null geodezik
olarak RK4 ile geriye integre edilir: kaçarsa yıldız alanı örneklenir,
düşerse gölge. Foton küresi 1.5, kritik çarpma parametresi sqrt(27)/2 ≈ 2.598 —
kapılar bu sayıları sayısal olarak yeniden üretir.
"""

import math

import numpy as np

RS = 1.0
R_SKY = 200.0          # yıldız küresi (kaçan ışın burada durur)
PHOTON_SPHERE = 1.5
B_CRIT = math.sqrt(27.0) / 2.0   # ≈ 2.598076


# ---------------------------------------------------------------- geodezik

def ray_u0(b, r0):
    """(du/dphi)² başlangıç değeri: 1/b² - u0²(1-u0), u0 = 1/r0."""
    u0 = 1.0 / r0
    return 1.0 / (b * b) - u0 * u0 * (1.0 - u0)


def integrate_u(b, r0, dphi=0.004, max_steps=4000, gr=True):
    """u'' = -u + 1.5u² RK4 (gr=False: u''=-u, düz uzay eşdeğeri).

    Dönüş: ('sky', phi_toplam) | ('capture', None). Kaçış eşiği r0 dışında.
    """
    r_sky = max(R_SKY, r0 * 1.15)
    u = 1.0 / r0
    du = math.sqrt(ray_u0(b, r0))
    phi = 0.0
    for _ in range(max_steps):
        k1u, k1d = du, (-u + (1.5 * u * u if gr else 0.0))
        k2u, k2d = du + 0.5 * dphi * k1d, -(u + 0.5 * dphi * k1u) + ((1.5 * (u + 0.5 * dphi * k1u) ** 2) if gr else 0.0)
        k3u, k3d = du + 0.5 * dphi * k2d, -(u + 0.5 * dphi * k2u) + ((1.5 * (u + 0.5 * dphi * k2u) ** 2) if gr else 0.0)
        k4u, k4d = du + dphi * k3d, -(u + dphi * k3u) + ((1.5 * (u + dphi * k3u) ** 2) if gr else 0.0)
        u_new = u + dphi / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        du_new = du + dphi / 6 * (k1d + 2 * k2d + 2 * k3d + k4d)
        phi += dphi
        if u_new >= 1.0 / RS:
            return ("capture", None)
        if u_new <= 1.0 / r_sky:
            # eşiği adım içinde nerede geçtiğini lineer geri al — kuantum hatası yok
            frac = (u - 1.0 / r_sky) / (u - u_new)
            return ("sky", phi - dphi + dphi * frac)
        u, du = u_new, du_new
    return ("capture", None)  # adım bitti: yakalanmış say (gölgeye yakın bant)


def deflection(b, r0=50.0, dphi=0.002):
    """Kaçan ışının GR bükülme açısı: GR süpürmesi - düz uzay süpürmesi.

    Fark almak sonlu r0/r_sky geometri payını otomatik siler — kalan saf
    kütleçekim bükülmesi (büyük b'de 2rs/b'ye yakınsar).
    """
    fate_gr, phi_gr = integrate_u(b, r0, dphi=dphi)
    if fate_gr != "sky":
        raise ValueError("b < b_crit: ışın kaçmıyor")
    fate_fl, phi_fl = integrate_u(b, r0, dphi=dphi, gr=False)
    return phi_gr - phi_fl


def b_crit_numeric(r0=50.0):
    """Kaç / düş bıçak ağzını ikili arama ile bul — sqrt(27)/2'ye karşı test edilir."""
    lo, hi = 1.0, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        fate, _ = integrate_u(mid, r0, dphi=0.002, max_steps=8000)
        if fate == "capture":
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------- yıldız alanı

def starfield(w=1024, h=512, n_stars=3800, seed=7, band=True):
    """Equirectangular yıldız haritası (RGB float HxWx3), tohumlu üretim.

    Yakın yıldızlar gauss splat, arka plan sönük; galaksi bandı büyük çemberde
    yumuşak parıltı. Kamera uzayında döndürülmez — BH haritası örneklecek.
    """
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 0.0015)
    ys, xs = np.mgrid[0:h, 0:w]

    # galaksi bandı: eğik büyük çember çevresinde geniş gauss
    tilt, shift = 0.55, rng.uniform(0, 2 * math.pi)
    lat = np.arcsin(np.clip((ys - h / 2) / (h / 2), -1, 1))
    lon = (xs / w) * 2 * math.pi
    d = np.arcsin(np.clip(np.sin(lat) * math.cos(tilt) +
                          np.cos(lat) * np.sin(tilt) * np.sin(lon - shift), -1, 1))
    band_glow = np.exp(-(d / 0.16) ** 2)
    img += band_glow[..., None] * np.array([0.05, 0.055, 0.075])

    for _ in range(n_stars):
        u = rng.uniform(-1, 1)
        theta = math.acos(u)                 # kutup açısı
        phi = rng.uniform(0, 2 * math.pi)
        cy = int(theta / math.pi * h) % h
        cx = int(phi / (2 * math.pi) * w) % w
        mag = rng.uniform(0.15, 1.0) ** 2.2
        temp = rng.uniform(0, 1)
        color = np.array([1.0,
                          0.75 + 0.25 * temp,
                          0.55 + 0.45 * temp]) if temp > 0.5 else \
            np.array([0.75 + 0.35 * temp, 0.82 + 0.15 * temp, 1.0])
        s = rng.uniform(0.45, 1.3)
        r = max(1, int(3 * s))
        for dy in range(-r, r + 1):
            yy = (cy + dy) % h
            for dx in range(-r, r + 1):
                g = math.exp(-(dx * dx + dy * dy) / (s * s))
                img[yy, (cx + dx) % w] += color * (mag * g)

    np.clip(img, 0.0, 4.0, out=img)
    return img


def sample_sky(img, directions):
    """Birim yön vektörlerinden (..., 3) yıldız alanı örneği (bilinear, equirect)."""
    x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
    lon = np.mod(np.arctan2(y, x), 2 * math.pi) / (2 * math.pi)
    lat = np.arcsin(np.clip(z, -1, 1)) / math.pi + 0.5
    h, w = img.shape[:2]
    fx = np.clip(lon * w, 0, w - 1.001)
    fy = np.clip(lat * h, 0, h - 1.001)
    x0 = fx.astype(int); y0 = fy.astype(int)
    tx = (fx - x0)[..., None]; ty = (fy - y0)[..., None]
    c = (img[y0, x0] * (1 - tx) * (1 - ty) +
         img[y0, x0 + 1] * tx * (1 - ty) +
         img[y0 + 1, x0] * (1 - tx) * ty +
         img[y0 + 1, x0 + 1] * tx * ty)
    return c


# ---------------------------------------------------------------- environment

def _integrate_batch(u, du, b, dphi=0.01):
    """Vektörel RK4: u'' = -u + 1.5u², tüm ışınlar paralel.

    Döner: (fate, phi) — fate: 0 kaçtı, 1 yakalandı, 2 tükendi.
    """
    n_steps = int(round(2.5 * math.pi / dphi))
    phi = np.zeros_like(u)
    alive = np.ones(len(u), dtype=bool)
    fate = np.full(len(u), 2, dtype=np.int8)
    for _ in range(n_steps):
        if not alive.any():
            break
        uu, ddu = u[alive], du[alive]
        k1u, k1d = ddu, -uu + 1.5 * uu * uu
        k2u, k2d = ddu + 0.5 * dphi * k1d, -(uu + 0.5 * dphi * k1u) + 1.5 * (uu + 0.5 * dphi * k1u) ** 2
        k3u, k3d = ddu + 0.5 * dphi * k2d, -(uu + 0.5 * dphi * k2u) + 1.5 * (uu + 0.5 * dphi * k2u) ** 2
        k4u, k4d = ddu + dphi * k3d, -(uu + dphi * k3u) + 1.5 * (uu + dphi * k3u) ** 2
        nu = uu + dphi / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        ndu = ddu + dphi / 6 * (k1d + 2 * k2d + 2 * k3d + k4d)
        u[alive], du[alive] = nu, ndu
        phi[alive] += dphi
        ai = np.where(alive)[0]
        cap = nu >= 1.0
        esc = (~cap) & (nu <= 1.0 / R_SKY) & (ndu < 0)
        done = cap | esc
        fate[ai[cap]] = 1
        fate[ai[esc]] = 0
        alive[ai[done]] = False
    return fate, phi


def lensed_env(r0=25.0, w=1024, h=512, sky=None, dphi=0.01):
    """BH'nin r0 uzaklığından görülen 360° lens haritası (HxWx3 RGB).

    Tüm pikseller aynı vektörel geodezikte: kaçan ışın, süpürdüğü toplam açı
    kadar döndürülmüş yönle yıldız alanını örnekler; u >= 1'e düşen gölgededir.
    """
    if sky is None:
        sky = starfield()
    lat = (np.arange(h) + 0.5) / h * math.pi - math.pi / 2
    lon = (np.arange(w) + 0.5) / w * 2 * math.pi
    lo, la = np.meshgrid(lon, lat)
    d = np.stack([np.cos(la) * np.sin(lo),
                  np.cos(la) * np.cos(lo),
                  np.sin(la)], axis=-1).reshape(-1, 3)

    rhat = np.array([0.0, 0.0, -1.0])
    up = np.array([1.0, 0.0, 0.0])
    cross = np.cross(np.broadcast_to(rhat, d.shape), d)
    sin_a = np.linalg.norm(cross, axis=-1)
    cos_a = -(d @ rhat)

    # çarpma parametresi: b = r0·sin(alpha) / sqrt(1 - 1/r0)
    b = r0 * sin_a / math.sqrt(1.0 - 1.0 / r0)
    u0 = 1.0 / r0
    rhs = np.maximum(1.0 / (b * b) - u0 * u0 * (1.0 - u0), 0.0)
    du0 = np.where(cos_a > 0.0, np.sqrt(rhs), -np.sqrt(rhs))  # içeri: u artar

    fate, phi = _integrate_batch(np.full(len(d), u0), du0, b, dphi=dphi)

    out = np.zeros((len(d), 3))
    esc = fate == 0
    if esc.any():
        plane_n = cross[esc]
        plane_n /= np.linalg.norm(plane_n, axis=-1)[:, None]
        dd = d[esc]
        c, s = np.cos(phi[esc])[:, None], np.sin(phi[esc])[:, None]
        rot = (c * dd + s * np.cross(plane_n, dd) +
               (1 - c) * np.sum(plane_n * dd, axis=-1)[:, None] * plane_n)
        rot /= np.linalg.norm(rot, axis=-1)[:, None]
        out[esc] = sample_sky(sky, rot)
    return out.reshape(h, w, 3)


# ---------------------------------------------------------------- akresyon diski

def disk_doppler(r, phi_pos, cam_dir):
    """Keplerian doppler beaming katsayısı (iyonize gazın ışıma şişmesi).

    r: yarıçap (rs), phi_pos: diski çember üzerinde konum açısı, cam_dir:
    kameraya birim yön (3). Dönüş: delta^3 — emissivity bunla çarpılır.
    beta = 1/sqrt(2(r - 1))  (halka içi sabit açısal momentum dönüşü, rs=1)
    """
    beta = 1.0 / math.sqrt(2.0 * (r - RS))
    beta = min(beta, 0.99)
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    # dönme yönü: teğet (-sin, cos); saat yönü sabit
    vel = np.array([-math.sin(phi_pos), math.cos(phi_pos), 0.0])
    mu = float(np.dot(vel, cam_dir))
    delta = 1.0 / (gamma * (1.0 - beta * mu))
    return delta ** 3


def disk_redshift(r):
    """Yerçekimi kızıla kayması çarpanı: g = sqrt(1 - 1/r)."""
    return math.sqrt(max(1.0 - RS / r, 0.0))
