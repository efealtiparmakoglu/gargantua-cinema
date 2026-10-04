"""
Gargantua — doğrulama kapıları.

    python3 tests/verify.py

Blender gerekmez (numpy var). Kritik soru: kendi entegratörümüz Einstein'ın
bilinen sayılarını yeniden üretiyor mu?
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import blackhole as bh  # noqa: E402

GATES = []


def gate(name):
    def deco(fn):
        GATES.append((name, fn))
        return fn
    return deco


@gate("G1 zayıf alan: sapma 2rs/b'ye uyor (Einstein 1915, rs=1 birimi)")
def g1_weak():
    errs = []
    for b in (50.0, 100.0, 200.0, 400.0):
        r0 = 8.0 * b                # b > r0 imkânsız; 8b sonlu-mesafe payını küçük tutar
        d = bh.deflection(b, r0=r0)
        want = 2.0 / b              # 4GM/c²b = 2rs/b, rs=1
        rel = abs(d - want) / want
        if rel > 0.10:
            errs.append(f"b={b}: sapma {d:.6f} != 2/b {want:.6f} (%{rel * 100:.1f})")
    return errs


@gate("G2 foton küresi: kritik çarpma parametresi sqrt(27)/2")
def g2_photon_sphere():
    errs = []
    got = bh.b_crit_numeric()
    want = math.sqrt(27.0) / 2.0
    rel = abs(got - want) / want
    if rel > 0.01:
        errs.append(f"b_crit sayısal {got:.6f} != teori {want:.6f} (%{rel * 100:.3f})")
    # foton küresi yarıçapının kendisi: b_crit = 3sqrt(3)/2 çemberi r=1.5'te kapanır
    if abs(bh.PHOTON_SPHERE - 1.5) > 1e-12:
        errs.append("foton küresi sabiti 1.5 değil")
    return errs


@gate("G3 gölge: b < b_crit düşer, b > b_crit kaçar (iki yönlü bıçak testi)")
def g3_shadow():
    errs = []
    for db in (-0.02, -0.005, 0.005, 0.02):
        b = bh.B_CRIT + db
        fate, _ = bh.integrate_u(b, r0=50.0, dphi=0.002, max_steps=9000)
        want = "capture" if db < 0 else "sky"
        if fate != want:
            errs.append(f"b={b:.4f} (b_crit{db:+.3f}): {fate} != {want}")
    return errs


@gate("G4 yıldız alanı: tohum tekrarlanabilir, yönler birim, tarama tutarlı")
def g4_sky():
    errs = []
    s1 = bh.starfield(seed=42, n_stars=60)
    s2 = bh.starfield(seed=42, n_stars=60)
    if not (s1.shape == s2.shape and abs(s1 - s2).max() < 1e-12):
        errs.append("aynı tohum farklı harita üretti")
    import numpy as np
    dirs = np.random.default_rng(1).normal(size=(64, 3))
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    c = bh.sample_sky(s1, dirs)
    if c.shape != (64, 3) or not np.isfinite(c).all():
        errs.append("sample_sky şekil/finite hatası")
    if c.max() <= 0.0:
        errs.append("örnekleme hep sıfır — yıldızlar eşlenmemiş olabilir")
    return errs


@gate("G5 doppler beaming: yaklaşan taraf parlar, kat sayı delta³")
def g5_doppler():
    errs = []
    r = 4.0
    # dönme vel=(-sinφ, cosφ); kamera +x'te -> yaklaşan nokta φ=-π/2
    near = bh.disk_doppler(r, -math.pi / 2, np.array([1.0, 0.0, 0.0]))
    far = bh.disk_doppler(r, math.pi / 2, np.array([1.0, 0.0, 0.0]))
    if not near > far * 2.0:
        errs.append(f"yaklaşan {near:.4f} uzaklaşanın 2 katı değil ({far:.4f})")
    beta = 1.0 / math.sqrt(2.0 * (r - 1.0))
    gamma = 1.0 / math.sqrt(1 - beta * beta)
    want_near = (1.0 / (gamma * (1 - beta))) ** 3
    if abs(near - want_near) > 1e-9:
        errs.append(f"delta³ {near:.9f} != formül {want_near:.9f}")
    # kızıla kayma: iç kenar daha sönük (g küçülür)
    if not bh.disk_redshift(2.0) < bh.disk_redshift(8.0):
        errs.append("iç kenar kızıla kayma çarpanı dıştan büyük olmamalı")
    return errs


@gate("G6 lens haritası: gölge kutupta, açısal yarıçapı b_crit geometrisiyle uyumlu")
def g6_env():
    w, h = 384, 192
    env = bh.lensed_env(r0=25.0, w=w, h=h, dphi=0.02)
    errs = []
    lum = env.mean(axis=2)
    row_max = lum.max(axis=1)           # gölge satırında TEK parlak piksel bile olamaz
    dark_rows = 0
    for i in range(h - 1, -1, -1):      # güney kutuptan yukarı say
        if row_max[i] < 0.01:
            dark_rows += 1
        else:
            break
    # gölge açısal yarıçapı: sin ψ = b_crit·sqrt(1-1/r0)/r0
    psi = math.asin(bh.B_CRIT * math.sqrt(1 - 1 / 25.0) / 25.0)
    want_rows = psi / math.pi * h
    if not 0.5 * want_rows <= dark_rows <= 1.8 * want_rows:
        errs.append(f"karanlık satır {dark_rows} != beklenen {want_rows:.1f} (ψ={math.degrees(psi):.2f}°)")
    if lum.max() < 0.2:
        errs.append("harita genelinde parlak yıldız yok — örnekleme çökmüş")
    return errs


def main():
    failures = 0
    for name, fn in GATES:
        errs = fn()
        if errs:
            failures += 1
            print(f"❌ {name}")
            for e in errs[:8]:
                print(f"   {e}")
        else:
            print(f"✅ {name}")
    print(f"\n{'KAPILAR KIRIK' if failures else 'TÜM KAPILAR YEŞİL'} ({failures} kırık)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
