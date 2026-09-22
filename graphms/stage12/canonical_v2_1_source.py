"""VERBATIM canonical Stage12 v2.1 backend source extracted from the frozen final Hybrid handoff notebook.

This file intentionally preserves notebook-level globals and is a provenance source,
not yet the standalone local wrapper. Do not alter scientific equations here.
"""

#@title Stage 12 backend definition — canonical v2.1 feature extractor { display-mode: "form" }
# Canonical Stage-12 v2.1 backend configuration preserved for the final Hybrid handoff.
SEG_THRESHOLD=.5; MIN_COMPONENT_VOXELS=5; CONNECTIVITY=26; FEATURE_SCHEMA_VERSION='GraphMSNet-Stage12-MAX-v2.1'; GLCM_Z_CLIP=5.; GLCM_BIN_WIDTH_Z=.25; GLCM_DISTANCES=(1,2); GLCM_MIN_VOXELS=8; ENABLE_ATLAS_LOCATION=True; FORCE_RECOMPUTE=False; CHECKPOINT_VERSION='graphms-v3.5.1-stage12-final-hybrid-v1'; CACHE_DIR=STAGE12_DEMO_CACHE; ATLAS_CACHE=STAGE12/'atlas_cache'; MODALITIES=['flair','t1','t2']; IMAGES_TR=RAW/'imagesTr'; IMAGES_TS=RAW/'imagesTs'
def sha256_small(path,chunk=1<<20):return sha256_file(path,chunk)
def case_from_mask_path(p):return p.name[:-7] if p.name.endswith('.nii.gz') else p.stem
def image_triplet(case_id,role):
    base=IMAGES_TR if role=='development_oof' else IMAGES_TS; return [base/f'{case_id}_{i:04d}.nii.gz' for i in range(3)]


# 3. Core feature engine: filter5, robust intensity, 3D shape, guide shape, sparse 3D GLCM.

STRUCT26 = ndimage.generate_binary_structure(3, 3)
EPS = 1e-12

def canonical_img(path):
    return nib.as_closest_canonical(nib.load(str(path)))

def filter5(mask):
    lab, n = ndimage.label(mask.astype(bool), structure=STRUCT26)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(lab.ravel())
    keep = np.where(sizes >= MIN_COMPONENT_VOXELS)[0]
    keep = keep[keep != 0]
    if len(keep) == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(lab, keep)

def robust_brain_scale(arr):
    a = np.asarray(arr, dtype=np.float32)
    brain = np.isfinite(a) & (np.abs(a) > EPS)
    vals = a[brain]
    if vals.size < 32:
        vals = a[np.isfinite(a)]
    if vals.size == 0:
        return 0.0, 1.0, brain
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    if mad < 1e-6:
        q25, q75 = np.percentile(vals, [25, 75])
        mad = float((q75 - q25) / 1.349)
    if mad < 1e-6:
        mad = float(np.std(vals))
    if mad < 1e-6:
        mad = 1.0
    return med, mad, brain

def first_order_features(values, prefix):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    keys = ["mean", "std", "median", "p10", "p25", "p75", "p90", "iqr", "skew", "kurtosis"]
    if v.size == 0:
        return {f"{prefix}_{k}": np.nan for k in keys}
    q10, q25, q50, q75, q90 = np.percentile(v, [10, 25, 50, 75, 90])
    return {
        f"{prefix}_mean": float(np.mean(v)),
        f"{prefix}_std": float(np.std(v)),
        f"{prefix}_median": float(q50),
        f"{prefix}_p10": float(q10),
        f"{prefix}_p25": float(q25),
        f"{prefix}_p75": float(q75),
        f"{prefix}_p90": float(q90),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_skew": float(stats.skew(v, bias=False)) if v.size >= 3 else np.nan,
        f"{prefix}_kurtosis": float(stats.kurtosis(v, fisher=True, bias=False)) if v.size >= 4 else np.nan,
    }

def glcm_directions(distance=1):
    # 13 unique half-space directions; symmetric accumulation represents all 26 neighbours.
    dirs = []
    for dx in range(-distance, distance + 1, distance):
        for dy in range(-distance, distance + 1, distance):
            for dz in range(-distance, distance + 1, distance):
                if dx == dy == dz == 0:
                    continue
                g = math.gcd(math.gcd(abs(dx), abs(dy)), abs(dz))
                if g != distance:
                    continue
                if (dx > 0) or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz > 0):
                    dirs.append((dx, dy, dz))
    return dirs


def _overlap_slices(shape, delta):
    src, dst = [], []
    for n, d in zip(shape, delta):
        if d >= 0:
            src.append(slice(0, n - d))
            dst.append(slice(d, n))
        else:
            src.append(slice(-d, n))
            dst.append(slice(0, n + d))
    return tuple(src), tuple(dst)

def sparse_glcm_features(zarr, mask, distance=1, prefix="flair_glcm_d1"):
    """Symmetric 3D lesion-ROI GLCM, vectorized for low wall-clock time."""
    names = ["contrast", "dissimilarity", "homogeneity", "asm", "energy",
             "correlation", "entropy", "pair_count"]
    if int(mask.sum()) < GLCM_MIN_VOXELS:
        return {f"{prefix}_{n}": (0.0 if n == "pair_count" else np.nan) for n in names}

    z = np.nan_to_num(
        np.clip(np.asarray(zarr, dtype=np.float32), -GLCM_Z_CLIP, GLCM_Z_CLIP),
        nan=0.0, posinf=GLCM_Z_CLIP, neginf=-GLCM_Z_CLIP
    )
    n_bins = int(round((2 * GLCM_Z_CLIP) / GLCM_BIN_WIDTH_Z)) + 1
    q = np.floor((z + GLCM_Z_CLIP) / GLCM_BIN_WIDTH_Z).astype(np.int16)
    q = np.clip(q, 0, n_bins - 1)

    counts = np.zeros(n_bins * n_bins, dtype=np.int64)
    pair_count = 0

    for delta in glcm_directions(distance):
        s0, s1 = _overlap_slices(mask.shape, delta)
        valid = mask[s0] & mask[s1]
        if not np.any(valid):
            continue
        a = q[s0][valid].astype(np.int64)
        b = q[s1][valid].astype(np.int64)
        counts += np.bincount(a * n_bins + b, minlength=n_bins*n_bins)
        counts += np.bincount(b * n_bins + a, minlength=n_bins*n_bins)
        pair_count += 2 * len(a)

    if pair_count == 0:
        return {f"{prefix}_{n}": (0.0 if n == "pair_count" else np.nan) for n in names}

    P = counts.reshape(n_bins, n_bins).astype(np.float64)
    P /= P.sum()
    I, J = np.indices(P.shape)
    diff = I - J

    contrast = float(np.sum(P * diff**2))
    dissimilarity = float(np.sum(P * np.abs(diff)))
    homogeneity = float(np.sum(P / (1.0 + diff**2)))
    asm = float(np.sum(P**2))
    energy = float(np.sqrt(asm))
    nz = P[P > 0]
    entropy = float(-np.sum(nz * np.log2(nz)))

    pi, pj = P.sum(axis=1), P.sum(axis=0)
    axis = np.arange(n_bins, dtype=float)
    mi, mj = float(np.sum(axis*pi)), float(np.sum(axis*pj))
    si = float(np.sqrt(np.sum(((axis-mi)**2)*pi)))
    sj = float(np.sqrt(np.sum(((axis-mj)**2)*pj)))
    corr = float(np.sum(P*(I-mi)*(J-mj))/(si*sj+EPS)) if si > 0 and sj > 0 else np.nan

    return {
        f"{prefix}_contrast": contrast,
        f"{prefix}_dissimilarity": dissimilarity,
        f"{prefix}_homogeneity": homogeneity,
        f"{prefix}_asm": asm,
        f"{prefix}_energy": energy,
        f"{prefix}_correlation": corr,
        f"{prefix}_entropy": entropy,
        f"{prefix}_pair_count": float(pair_count),
    }

def surface_area_mm2(mask, spacing):
    if int(mask.sum()) < 4:
        return np.nan
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(axis=0) - 1, 0)
    hi = np.minimum(idx.max(axis=0) + 2, np.array(mask.shape))
    crop = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.uint8)
    if min(crop.shape) < 2:
        return np.nan
    try:
        verts, faces, _, _ = measure.marching_cubes(crop, level=0.5, spacing=spacing)
        return float(measure.mesh_surface_area(verts, faces))
    except Exception:
        return np.nan

def guide_2d_shape(mask, spacing):
    if not mask.any():
        return {
            "guide_max_slice_area_mm2": 0.0,
            "guide_max_slice_perimeter_mm": 0.0,
            "guide_max_slice_compactness": np.nan,
        }
    sx, sy, _ = spacing
    areas = mask.sum(axis=(0, 1))
    z = int(np.argmax(areas))
    sl = mask[:, :, z]
    area = float(sl.sum() * sx * sy)
    per_px = float(measure.perimeter_crofton(sl.astype(np.uint8), directions=4))
    perimeter = per_px * float((sx + sy) / 2.0)
    compact = float(4.0 * math.pi * area / (perimeter**2 + EPS)) if perimeter > 0 else np.nan
    return {
        "guide_max_slice_area_mm2": area,
        "guide_max_slice_perimeter_mm": perimeter,
        "guide_max_slice_compactness": compact,
    }

def pca_shape(mask, spacing):
    coords = np.argwhere(mask)
    if len(coords) < 4:
        return {"shape_elongation": np.nan, "shape_flatness": np.nan}
    pts = coords.astype(np.float64) * np.asarray(spacing)[None, :]
    if len(pts) > 100000:
        take = np.linspace(0, len(pts) - 1, 100000).astype(int)
        pts = pts[take]
    ev = np.linalg.eigvalsh(np.cov(pts.T))
    ev = np.sort(np.maximum(ev, 0))[::-1]
    if ev[0] <= EPS:
        return {"shape_elongation": np.nan, "shape_flatness": np.nan}
    return {
        "shape_elongation": float(np.sqrt(ev[1] / ev[0])) if len(ev) > 1 else np.nan,
        "shape_flatness": float(np.sqrt(ev[2] / ev[0])) if len(ev) > 2 else np.nan,
    }

print("Core feature engine ready.")


# 4. Optional atlas / periventricular context.
# Highest-value extension, but fail-open: core Stage 12 must still finish if atlas download/API fails.
ATLAS = {"enabled": False, "reason": "disabled"}

def ensure_nilearn():
    import importlib.util, subprocess, sys
    if importlib.util.find_spec("nilearn") is None:
        print("Installing nilearn for atlas features...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "nilearn"])

def as_img(x):
    from nilearn.image import load_img
    return load_img(x)

def normalise_label_name(x):
    return re.sub(r"[^a-z0-9]+", " ", str(x).lower()).strip()

if ENABLE_ATLAS_LOCATION:
    try:
        ensure_nilearn()
        from nilearn.datasets import fetch_atlas_harvard_oxford

        cort = fetch_atlas_harvard_oxford(
            "cort-maxprob-thr25-2mm", data_dir=str(ATLAS_CACHE), symmetric_split=False
        )
        sub = fetch_atlas_harvard_oxford(
            "sub-maxprob-thr25-2mm", data_dir=str(ATLAS_CACHE), symmetric_split=False
        )

        cort_img = as_img(cort.maps)
        sub_img = as_img(sub.maps)
        cort_arr = np.asarray(cort_img.dataobj)
        sub_arr = np.asarray(sub_img.dataobj)
        cort_aff = np.asarray(cort_img.affine)
        sub_aff = np.asarray(sub_img.affine)

        sub_labels = [normalise_label_name(x) for x in sub.labels]
        vent_ids = [
            i for i, name in enumerate(sub_labels)
            if ("lateral ventricle" in name) or ("inferior lateral ventricle" in name)
        ]
        brainstem_ids = [i for i, name in enumerate(sub_labels) if ("brain stem" in name or "brainstem" in name)]

        vent_mask = np.isin(sub_arr, vent_ids) if vent_ids else np.zeros_like(sub_arr, dtype=bool)
        sub_zooms = sub_img.header.get_zooms()[:3]
        vent_dist = ndimage.distance_transform_edt(~vent_mask, sampling=sub_zooms) if vent_mask.any() else None

        ATLAS = {
            "enabled": True,
            "cort_arr": cort_arr,
            "sub_arr": sub_arr,
            "cort_aff": cort_aff,
            "sub_aff": sub_aff,
            "cort_labels": list(cort.labels),
            "sub_labels": list(sub.labels),
            "brainstem_ids": brainstem_ids,
            "vent_dist": vent_dist,
            "source": "Harvard-Oxford maxprob-thr25-2mm",
        }
        print("Atlas features enabled:", ATLAS["source"])
    except Exception as e:
        ATLAS = {"enabled": False, "reason": repr(e)}
        print("WARNING: atlas features disabled; core Stage 12 will continue.")
        print("Reason:", repr(e))
else:
    print("Atlas features disabled by configuration.")

def sample_world_array(world_xyz, arr, affine):
    if len(world_xyz) == 0:
        return np.array([], dtype=arr.dtype), np.array([], dtype=bool)
    inv = np.linalg.inv(affine)
    hom = np.c_[world_xyz, np.ones(len(world_xyz))]
    ijk = np.rint((inv @ hom.T).T[:, :3]).astype(int)
    valid = np.all((ijk >= 0) & (ijk < np.asarray(arr.shape)[None, :]), axis=1)
    vals = np.zeros(len(ijk), dtype=arr.dtype)
    v = ijk[valid]
    if len(v):
        vals[valid] = arr[v[:, 0], v[:, 1], v[:, 2]]
    return vals, valid

def atlas_features(mask, mask_affine, voxel_volume_mm3):
    base = {
        "atlas_inbounds_fraction": np.nan,
        "atlas_cortical_overlap_mm3": np.nan,
        "atlas_subcortical_overlap_mm3": np.nan,
        "atlas_brainstem_overlap_mm3": np.nan,
        "periventricular_fraction_le3mm": np.nan,
        "periventricular_fraction_le5mm": np.nan,
        "periventricular_fraction_le10mm": np.nan,
        "ventricle_distance_mean_mm": np.nan,
        "ventricle_distance_min_mm": np.nan,
    }
    if not ATLAS.get("enabled") or not mask.any():
        return base

    vox = np.argwhere(mask)
    hom = np.c_[vox, np.ones(len(vox))]
    world = (mask_affine @ hom.T).T[:, :3]

    cort_vals, cort_valid = sample_world_array(world, ATLAS["cort_arr"], ATLAS["cort_aff"])
    sub_vals, sub_valid = sample_world_array(world, ATLAS["sub_arr"], ATLAS["sub_aff"])
    valid = cort_valid & sub_valid
    inb = float(valid.mean()) if len(valid) else 0.0
    base["atlas_inbounds_fraction"] = inb

    # Guard against silently sampling a clearly incompatible coordinate system.
    if inb < 0.80:
        return base

    c = cort_vals[valid]
    s = sub_vals[valid]
    base["atlas_cortical_overlap_mm3"] = float(np.sum(c > 0) * voxel_volume_mm3)
    base["atlas_subcortical_overlap_mm3"] = float(np.sum(s > 0) * voxel_volume_mm3)
    if ATLAS["brainstem_ids"]:
        base["atlas_brainstem_overlap_mm3"] = float(
            np.sum(np.isin(s, ATLAS["brainstem_ids"])) * voxel_volume_mm3
        )

    if ATLAS.get("vent_dist") is not None:
        dvals, dvalid = sample_world_array(world, ATLAS["vent_dist"], ATLAS["sub_aff"])
        d = dvals[dvalid].astype(float)
        if len(d):
            base["periventricular_fraction_le3mm"] = float(np.mean(d <= 3.0))
            base["periventricular_fraction_le5mm"] = float(np.mean(d <= 5.0))
            base["periventricular_fraction_le10mm"] = float(np.mean(d <= 10.0))
            base["ventricle_distance_mean_mm"] = float(np.mean(d))
            base["ventricle_distance_min_mm"] = float(np.min(d))
    return base


# 5. Case-level + lesion-level extractor.
MODALITIES = ["flair", "t1", "t2"]

def assert_same_geometry(a, b, name_a="mask", name_b="image"):
    if a.shape != b.shape:
        raise ValueError(f"Geometry mismatch {name_a}{a.shape} vs {name_b}{b.shape}")
    if not np.allclose(a.affine, b.affine, atol=1e-3, rtol=0):
        raise ValueError(f"Affine mismatch between {name_a} and {name_b}")

def parse_case_id(case_id):
    m = re.match(r"^MSLesSeg_(P\d+)(?:_(T\d+))?$", case_id)
    if not m:
        raise ValueError(f"Unexpected case id: {case_id}")
    patient = m.group(1)
    timepoint = m.group(2) or "T1"
    return patient, timepoint

def lesion_component_table(case_id, role, mask, affine, spacing):
    lab, n = ndimage.label(mask, structure=STRUCT26)
    voxel_vol = float(np.prod(spacing))
    rows = []
    for k in range(1, n + 1):
        idx = np.argwhere(lab == k)
        if len(idx) == 0:
            continue
        hom = np.c_[idx, np.ones(len(idx))]
        world = (affine @ hom.T).T[:, :3]
        lo = idx.min(axis=0); hi = idx.max(axis=0) + 1
        rows.append({
            "case_id": case_id,
            "role": role,
            "lesion_id": int(k),
            "lesion_voxels": int(len(idx)),
            "lesion_volume_mm3": float(len(idx) * voxel_vol),
            "centroid_world_x_mm": float(world[:, 0].mean()),
            "centroid_world_y_mm": float(world[:, 1].mean()),
            "centroid_world_z_mm": float(world[:, 2].mean()),
            "bbox_x_mm": float((hi[0] - lo[0]) * spacing[0]),
            "bbox_y_mm": float((hi[1] - lo[1]) * spacing[1]),
            "bbox_z_mm": float((hi[2] - lo[2]) * spacing[2]),
        })
    return rows


def atomic_write_text(path, text):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)

def source_stat_signature(path):
    st = Path(path).stat()
    return {"name": Path(path).name, "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}

def feature_config_fingerprint():
    payload = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "seg_threshold": SEG_THRESHOLD,
        "min_component_voxels": MIN_COMPONENT_VOXELS,
        "connectivity": CONNECTIVITY,
        "glcm_z_clip": GLCM_Z_CLIP,
        "glcm_bin_width_z": GLCM_BIN_WIDTH_Z,
        "glcm_distances": list(GLCM_DISTANCES),
        "glcm_min_voxels": GLCM_MIN_VOXELS,
        "atlas_enabled_runtime": bool(ATLAS.get("enabled")),
        "atlas_source": ATLAS.get("source"),
        "winner_recipe_sha256": sha256_small(WINNING_RECIPE),
    }
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()

CONFIG_FINGERPRINT = feature_config_fingerprint()

def case_input_fingerprint(mask_path, role):
    case_id = case_from_mask_path(mask_path)
    payload = {
        "case_id": case_id,
        "role": role,
        "config": CONFIG_FINGERPRINT,
        "mask_sha256": sha256_small(mask_path),
        "mri_sources": [source_stat_signature(p) for p in image_triplet(case_id, role)],
    }
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def case_cache_paths(case_id):
    return CACHE_DIR / f"{case_id}.json", CACHE_DIR / f"{case_id}_lesions.json"

def extract_case(mask_path, role):
    case_id = case_from_mask_path(mask_path)
    cpath, lpath = case_cache_paths(case_id)
    input_fp = case_input_fingerprint(mask_path, role)
    if cpath.exists() and lpath.exists() and not FORCE_RECOMPUTE:
        rec = json.loads(cpath.read_text())
        if (
            rec.get("_feature_schema_version") == FEATURE_SCHEMA_VERSION
            and rec.get("_input_fingerprint") == input_fp
            and rec.get("_config_fingerprint") == CONFIG_FINGERPRINT
        ):
            return rec, json.loads(lpath.read_text())

    t0 = time.perf_counter()
    patient, timepoint = parse_case_id(case_id)

    mask_img = canonical_img(mask_path)
    mask_prob_or_label = np.asarray(mask_img.dataobj)
    # CV/test folders contain labels; thresholding is harmless and keeps the contract explicit.
    mask = filter5(mask_prob_or_label >= SEG_THRESHOLD)
    spacing = tuple(float(x) for x in mask_img.header.get_zooms()[:3])
    voxel_vol = float(np.prod(spacing))
    lesion_voxels = int(mask.sum())
    lesion_volume_mm3 = float(lesion_voxels * voxel_vol)

    lab, n_comp = ndimage.label(mask, structure=STRUCT26)
    sizes_vox = np.bincount(lab.ravel())[1:] if n_comp else np.array([], dtype=int)
    sizes_mm3 = sizes_vox.astype(float) * voxel_vol
    sort_sizes = np.sort(sizes_mm3)[::-1] if len(sizes_mm3) else np.array([], dtype=float)

    rec = {
        "_feature_schema_version": FEATURE_SCHEMA_VERSION,
        "_input_fingerprint": input_fp,
        "_config_fingerprint": CONFIG_FINGERPRINT,
        "case_id": case_id,
        "patient_id": patient,
        "timepoint": timepoint,
        "role": role,
        "mask_source": str(mask_path),
        "frozen_segmentation": "GraphMS v3.5.1 Hybrid + cross-fitted Stage11-v1 + 26-connectivity + no morphology",
        "ground_truth_mask_used": False,
        "lesion_voxels": lesion_voxels,
        "lesion_volume_mm3": lesion_volume_mm3,
        "lesion_volume_ml": lesion_volume_mm3 / 1000.0,
        "log_lesion_volume": float(np.log1p(lesion_volume_mm3)),
        "lesion_count": int(n_comp),
        "largest_lesion_volume_mm3": float(sort_sizes[0]) if len(sort_sizes) else 0.0,
        "second_largest_lesion_volume_mm3": float(sort_sizes[1]) if len(sort_sizes) > 1 else 0.0,
        "mean_lesion_volume_mm3": float(np.mean(sizes_mm3)) if len(sizes_mm3) else 0.0,
        "median_lesion_volume_mm3": float(np.median(sizes_mm3)) if len(sizes_mm3) else 0.0,
        "std_lesion_volume_mm3": float(np.std(sizes_mm3)) if len(sizes_mm3) else 0.0,
        "largest_lesion_fraction": float(sort_sizes[0] / lesion_volume_mm3) if lesion_volume_mm3 > 0 else 0.0,
        "lesion_count_lt50mm3": int(np.sum(sizes_mm3 < 50.0)),
        "lesion_count_50_200mm3": int(np.sum((sizes_mm3 >= 50.0) & (sizes_mm3 < 200.0))),
        "lesion_count_ge200mm3": int(np.sum(sizes_mm3 >= 200.0)),
        "voxel_volume_mm3": voxel_vol,
        "spacing_x_mm": spacing[0],
        "spacing_y_mm": spacing[1],
        "spacing_z_mm": spacing[2],
    }

    # Global 3D morphology.
    area = surface_area_mm2(mask, spacing)
    rec["shape_surface_area_mm2"] = area
    rec["shape_sphericity"] = (
        float((math.pi ** (1/3)) * ((6.0 * lesion_volume_mm3) ** (2/3)) / (area + EPS))
        if lesion_volume_mm3 > 0 and np.isfinite(area) and area > 0 else np.nan
    )
    rec["shape_compactness_3d"] = (
        float(36.0 * math.pi * lesion_volume_mm3**2 / (area**3 + EPS))
        if lesion_volume_mm3 > 0 and np.isfinite(area) and area > 0 else np.nan
    )
    rec.update(guide_2d_shape(mask, spacing))
    rec.update(pca_shape(mask, spacing))

    # Spatial burden in world coordinates. These are cheap and useful even without an atlas.
    vox = np.argwhere(mask)
    if len(vox):
        hom = np.c_[vox, np.ones(len(vox))]
        world = (mask_img.affine @ hom.T).T[:, :3]
        rec.update({
            "lesion_centroid_world_x_mm": float(world[:, 0].mean()),
            "lesion_centroid_world_y_mm": float(world[:, 1].mean()),
            "lesion_centroid_world_z_mm": float(world[:, 2].mean()),
            "lesion_fraction_left_xlt0": float(np.mean(world[:, 0] < 0)),
            "lesion_fraction_right_xge0": float(np.mean(world[:, 0] >= 0)),
            "lesion_world_x_std_mm": float(world[:, 0].std()),
            "lesion_world_y_std_mm": float(world[:, 1].std()),
            "lesion_world_z_std_mm": float(world[:, 2].std()),
        })
    else:
        for k in [
            "lesion_centroid_world_x_mm","lesion_centroid_world_y_mm","lesion_centroid_world_z_mm",
            "lesion_fraction_left_xlt0","lesion_fraction_right_xge0",
            "lesion_world_x_std_mm","lesion_world_y_std_mm","lesion_world_z_std_mm"
        ]:
            rec[k] = np.nan

    rec.update(atlas_features(mask, mask_img.affine, voxel_vol))

    # MRI features. Load one modality at a time to limit RAM.
    triplet = image_triplet(case_id, role)
    approx_brain_volume_mm3 = np.nan

    for mod, img_path in zip(MODALITIES, triplet):
        img = canonical_img(img_path)
        assert_same_geometry(mask_img, img, "mask", mod)
        arr = np.asarray(img.dataobj, dtype=np.float32)
        med, scale, brain = robust_brain_scale(arr)
        zarr = (arr - med) / scale

        lesion_raw = arr[mask]
        lesion_z = zarr[mask]
        rec.update(first_order_features(lesion_raw, f"{mod}_raw"))
        rec.update(first_order_features(lesion_z, f"{mod}_z"))

        rec[f"{mod}_brain_center"] = med
        rec[f"{mod}_brain_robust_scale"] = scale

        for d in GLCM_DISTANCES:
            rec.update(
                sparse_glcm_features(
                    zarr, mask, distance=d, prefix=f"{mod}_glcm_d{d}"
                )
            )

        if mod == "flair":
            approx_brain_volume_mm3 = float(brain.sum() * voxel_vol)

        del arr, zarr

    rec["approx_brain_volume_mm3"] = approx_brain_volume_mm3
    rec["lesion_volume_normalized"] = (
        float(lesion_volume_mm3 / approx_brain_volume_mm3)
        if np.isfinite(approx_brain_volume_mm3) and approx_brain_volume_mm3 > 0 else np.nan
    )

    lesions = lesion_component_table(case_id, role, mask, mask_img.affine, spacing)
    rec["runtime_sec"] = float(time.perf_counter() - t0)

    # JSON-safe cache.
    clean = {}
    for k, v in rec.items():
        if isinstance(v, (np.floating, float)):
            clean[k] = None if not np.isfinite(v) else float(v)
        elif isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (bool, np.bool_)):
            clean[k] = bool(v)
        else:
            clean[k] = v
    atomic_write_text(cpath, json.dumps(clean, indent=2))
    atomic_write_text(lpath, json.dumps(lesions, indent=2))
    return clean, lesions

print("Case extractor ready.")
