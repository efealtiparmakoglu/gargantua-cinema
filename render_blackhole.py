"""
Gargantua sinematik: lenslenmiş yıldız alanı + gerçek doppler'lı akresyon diski.

    blender --background --python render_blackhole.py -- --preview   # 4 kare 640x360
    blender --background --python render_blackhole.py               # 48 kare 960x540
"""

import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import bpy  # noqa: E402

import blackhole as bh  # noqa: E402

R_IN, R_OUT = 3.0, 14.0     # disk: ISCO (3rs) -> 14rs
SHADOW_R = 2.0             # merkezi karanlık küre (env gölgesiyle birleşir)


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return {"preview": "--preview" in a}


def _lineer(ob):
    act = ob.animation_data.action
    if hasattr(act, "fcurves"):
        fcs = list(act.fcurves)
    else:
        fcs = [fc for layer in act.layers for strip in layer.strips
               for bag in strip.channelbags for fc in bag.fcurves]
    for fc in fcs:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def env_dunyasi(preview):
    """Lens haritasını üret, Blender imajına doldur, world env olarak bağla."""
    w, h = (1024, 512) if preview else (2048, 1024)
    sky = bh.starfield(seed=7, n_stars=2600)
    env = bh.lensed_env(r0=25.0, w=w, h=h, sky=sky, dphi=0.01)
    env = np.clip(env, 0, 1) ** 0.85   # hafif ton açma (AgX öncesi)

    img = bpy.data.images.new("LensEnv", w, h, alpha=False)
    img.colorspace_settings.name = "Linear Rec.709"
    rgba = np.ones((h, w, 4))
    rgba[:, :, :3] = env
    img.pixels.foreach_set(rgba.reshape(-1).astype(np.float32))

    world = bpy.data.worlds.new("Kara")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 1.6
    tex = nt.nodes.new("ShaderNodeTexEnvironment")
    tex.image = img
    nt.links.new(tex.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def halka_dokusu():
    """Disk dokusu: radyal sıcaklık rampası + diferansiyel kayma çizikleri (numpy)."""
    wa, wr = 1024, 128   # açı × yarıçap
    rng = np.random.default_rng(3)
    g = rng.uniform(0.35, 1.0, (24, 12))
    gi = np.zeros((wa, wr, 3))
    ph = np.linspace(0, 2 * math.pi, wa, endpoint=False)
    rr = np.linspace(0, 1, wr)
    # değer gürültüsü (açıda doğal döngü: kosinüs interpolasyonlu grid)
    for k in range(4):
        n = rng.uniform(0, 1, (12, 8))
        phx = np.linspace(0, 12, wa, endpoint=False)
        ryx = np.linspace(0, 8, wr, endpoint=False)
        n_bi = np.zeros((wa, wr))
        for i in range(wa):
            i0, f = int(phx[i]), phx[i] - int(phx[i])
            i1 = (i0 + 1) % 12
            for j in range(wr):
                j0, g2 = int(ryx[j]), ryx[j] - int(ryx[j])
                j1 = min(j0 + 1, 7)
                a = n[i0, j0] * (1 - f) + n[i1, j0] * f
                b2 = n[i0, j1] * (1 - f) + n[i1, j1] * f
                n_bi[i, j] = a * (1 - g2) + b2 * g2
        scale = 0.5 / (k + 1)
        gi += n_bi[..., None] * scale
    gi /= gi.max()

    r = np.linspace(R_IN, R_OUT, wr)
    t = (r - R_IN) / (R_OUT - R_IN)
    temp = (1 - t) ** 1.35                               # sıcak -> soğuk (dış disk doppler'ı görünür kalsın)
    streak = 0.72 + 0.28 * np.sin(ph[:, None] * 6 + rr[None, :] * 14 + gi[:, :, 0] * 9)
    tex = temp[None, :, None] * streak[..., None]
    # renk: iç beyaz-sarı -> dış derin turuncu-kızıl
    hot = np.array([1.00, 0.93, 0.78])
    mid = np.array([1.00, 0.55, 0.18])
    cold = np.array([0.45, 0.10, 0.02])
    col = np.zeros((wa, wr, 3))
    m1 = t < 0.35
    f1 = np.clip(t / 0.35, 0, 1)
    m2 = ~m1
    f2 = np.clip((t - 0.35) / 0.65, 0, 1)
    for j in range(wr):
        cj = hot * (1 - f1[j]) + mid * f1[j] if f1[j] < 1 or True else mid
        cj2 = mid * (1 - f2[j]) + cold * f2[j]
        col[:, j, :] = (cj if t[j] < 0.35 else cj2)
    return np.clip(tex * col * 2.6, 0, 12)


def disk_mesh(preview):
    """Annulus + UV + PIŞİRİLMİŞ doppler beaming dokusu (numpy, node-API'siz).

    Beaming, klip ortası kamera yönü için hesaplanır (azimut süpürmesi 32° —
    sapma görsel olarak marjinal). Fizik aynı: delta³ · g · sıcaklık rampası.
    """
    seg, rings = 160, 10
    verts, faces, uvs = [], [], []
    for ri in range(rings + 1):
        r = R_IN + (R_OUT - R_IN) * ri / rings
        for si in range(seg):
            a = 2 * math.pi * si / seg
            verts.append((r * math.cos(a), r * math.sin(a), 0.0))
            uvs.append((si / seg, ri / rings))
    for ri in range(rings):
        for si in range(seg):
            a = ri * seg + si
            b2 = ri * seg + (si + 1) % seg
            c = (ri + 1) * seg + (si + 1) % seg
            d = (ri + 1) * seg + si
            faces.append((a, b2, c, d))
    mesh = bpy.data.meshes.new("Disk")
    mesh.from_pydata(verts, [], faces)
    uv = mesh.uv_layers.new()
    # UV PER-FACE: sarmal yüzün si=0 köşeleri u=1.0 alir (0.994->0 ters örneklem yok)
    for fi, (ri, si) in enumerate((a_ for a_ in [(r_, s_) for r_ in range(rings) for s_ in range(seg)])):
        pass  # yer tutucu; asagida dogrudan loop bazli yazilir
    face_si = [(r_, s_) for r_ in range(rings) for s_ in range(seg)]
    for poly in mesh.polygons:
        fi = poly.index
        r_, s_ = face_si[fi]
        for li in poly.loop_indices:
            vi = mesh.loops[li].vertex_index
            ring_i, seg_i = divmod(vi, seg)
            u = seg_i / seg
            if seg_i == 0 and s_ == seg - 1:
                u = 1.0                      # sarmal yüzün bitiş köşesi
            uv.data[li].uv = (u, ring_i / rings)
    ob = bpy.data.objects.new("AkresyonDiski", mesh)
    bpy.context.scene.collection.objects.link(ob)

    # ---- pişirme: (açı 1024, yarıçap 128) dokusuna tüm fizik
    wa, wr = 1024, 128
    ph = np.linspace(0, 2 * math.pi, wa, endpoint=False)
    rr = np.linspace(R_IN, R_OUT, wr)
    PH, RR = np.meshgrid(ph, rr, indexing="ij")

    # klip ortası kamera: azimut 16°, yükseklik ~11°, uzaklık ~26rs
    ca, ce = math.radians(16), math.radians(11)
    cam = 26.0 * np.array([math.cos(ce) * math.cos(ca), math.cos(ce) * math.sin(ca), math.sin(ce)])
    pos = np.stack([RR * np.cos(PH), RR * np.sin(PH), np.zeros_like(PH)], axis=-1)
    cdir = cam[None, None, :] - pos
    cdir /= np.linalg.norm(cdir, axis=-1, keepdims=True)

    beta = 1.0 / np.sqrt(2.0 * (RR - 1.0))
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    vel = np.stack([-np.sin(PH), np.cos(PH), np.zeros_like(PH)], axis=-1)
    mu = np.sum(vel * cdir, axis=-1)
    delta3 = (1.0 / (gamma * (1.0 - beta * mu))) ** 3
    gshift = np.sqrt(np.maximum(1.0 - 1.0 / RR, 0.0))

    tex = halka_dokusu()                       # (wa, wr, 3) sıcaklık rampası + çizikler
    baked = np.clip(tex * (delta3 * gshift)[..., None], 0.0, 5.0)
    # açısal dikişi öldür: wrap'li [1,2,1]/4 yumuşatma, iki tur
    for _ in range(2):
        baked = (np.roll(baked, 1, axis=0) + 2 * baked + np.roll(baked, -1, axis=0)) / 4.0

    img = bpy.data.images.new("DiskBaked", wa, wr, alpha=False)
    img.colorspace_settings.name = "Non-Color"
    rgba = np.ones((wr, wa, 4))
    rgba[:, :, :3] = baked.transpose(1, 0, 2)
    img.pixels.foreach_set(np.flipud(rgba).reshape(-1).astype(np.float32))

    m = bpy.data.materials.new("DiskMat")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 1.0
    texn = nt.nodes.new("ShaderNodeTexImage")
    texn.image = img
    nt.links.new(texn.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    ob.data.materials.append(m)
    return ob


def main():
    opts = args()
    sc = bpy.context.scene
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)

    env_dunyasi(opts["preview"])

    # gölge küresi: tam siyah, env'in merkezi gölgesiyle birleşir
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=SHADOW_R)
    hole = bpy.context.active_object
    hole.name = "Ufuk"
    hm = bpy.data.materials.new("Siyah")
    hm.use_nodes = True
    hb = hm.node_tree.nodes["Principled BSDF"]
    hb.inputs["Base Color"].default_value = (0, 0, 0, 1)
    hb.inputs["Roughness"].default_value = 1.0
    hole.data.materials.append(hm)

    disk_mesh(opts["preview"])

    # kamera: geniş üst açıdan diskin düzlemine iner, azimut yavaşça süpürür
    cam = bpy.data.cameras.new("Cam")
    cam.lens = 35
    co = bpy.data.objects.new("Kamera", cam)
    bpy.context.scene.collection.objects.link(co)
    hedef = bpy.data.objects.new("Hedef", None)
    bpy.context.scene.collection.objects.link(hedef)
    co.constraints.new("TRACK_TO").target = hedef
    hedef.location = (0, 0, 0.4)

    def kamera_konumu(t, r, elev_deg, az_deg):
        e, a = math.radians(elev_deg), math.radians(az_deg)
        return (r * math.cos(e) * math.cos(a),
                r * math.cos(e) * math.sin(a),
                r * math.sin(e) + 0.4)

    f0, f1 = 1, 49
    co.location = kamera_konumu(0, 40, 20, 0)
    co.keyframe_insert("location", frame=f0)
    co.location = kamera_konumu(1, 13.5, 6.5, 32)
    co.keyframe_insert("location", frame=f1)
    hedef.keyframe_insert("location", frame=f0)
    hedef.location = (0, 0, 0.0)
    hedef.keyframe_insert("location", frame=f1)
    _lineer(co)
    _lineer(hedef)
    sc.camera = co

    # render ayarları
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    sc.cycles.device = "GPU"
    if opts["preview"]:
        sc.render.resolution_x, sc.render.resolution_y = 640, 360
        sc.cycles.samples = 32
    else:
        sc.render.resolution_x, sc.render.resolution_y = 960, 540
        sc.cycles.samples = 96
    sc.cycles.use_denoising = False
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.view_settings.exposure = 0.2
    sc.render.image_settings.file_format = "PNG"

    # kompositör: fog glow (diske sinematik parıltı) — 5.x API farkında sessizce atlanır
    try:
        sc.use_nodes = True
        nt = sc.node_tree
        nt.nodes.clear()
        rl = nt.nodes.new("CompositorNodeRLayers")
        gl = nt.nodes.new("CompositorNodeGlare")
        gl.glare_type = "FOG_GLOW"
        gl.quality = "MEDIUM"
        gl.size = 8
        comp = nt.nodes.new("CompositorNodeComposite")
        nt.links.new(rl.outputs["Image"], gl.inputs["Image"])
        nt.links.new(gl.outputs["Image"], comp.inputs["Image"])
    except Exception as e:  # noqa: BLE001
        print(f"  [atlandı] compositor bloom: {e}")

    out = os.path.join(REPO, "renders", "preview" if opts["preview"] else "gargantua")
    os.makedirs(out, exist_ok=True)
    frames = [1, 16, 32, 49] if opts["preview"] else range(1, 50)
    for f in frames:
        sc.frame_set(f)
        sc.render.filepath = os.path.join(out, f"frame_{f:04d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"[kare {f}] bitti", flush=True)
    print(f"== BİTTİ -> {out} ==")


main()
