
import csv, json, os, re, sys, math, time, statistics
from collections import defaultdict
import numpy as np

# ----- paths (yours) -----
CATALOG = os.environ.get("CATALOG", "/netdisk/ly6/GSAS_Projects/Impurity_detection_GSAS_ver2/data/database_aug/catalog.csv")
METAJS  = os.environ.get("METAJS",  "/netdisk/ly6/GSAS_Projects/Impurity_detection_GSAS_ver2/data/database_aug/highsymm_metadata.json")
INDEX   = os.environ.get("INDEX",   "/netdisk/ly6/GSAS_Projects/Impurity_detection_GSAS_ver2/data/database_aug/profiles64/index.csv")
NPZ     = os.environ.get("NPZ",     "/netdisk/ly6/GSAS_Projects/Impurity_detection_GSAS_ver2/data/database_aug/profiles64/profiles64.npz")

# ===================== Config =====================
PCT = float(os.environ.get("PCT", "5"))  # percent tol for a,b,c,alpha,beta,gamma
PCT_VOLUME = 3.0 * PCT                   # <-- CHANGED: V uses 3×PCT
TS  = time.strftime("%Y%m%d-%H%M%S")

CAT_DIR = os.path.dirname(CATALOG)
IDX_DIR = os.path.dirname(INDEX)
NPZ_DIR = os.path.dirname(NPZ)

OUT_CAT  = os.path.join(CAT_DIR, f"catalog.deduplicated.{TS}.csv")
OUT_LOG  = os.path.join(CAT_DIR, f"dedupe_log.{TS}.csv")
OUT_SUM  = os.path.join(CAT_DIR, f"dedupe_summary.{TS}.txt")

OUT_INDEX_DROPSAFE = os.path.join(IDX_DIR, "index_deduplicated.csv")
if os.path.exists(OUT_INDEX_DROPSAFE):
    OUT_INDEX_DROPSAFE = os.path.join(IDX_DIR, f"index_deduplicated.{TS}.csv")

OUT_NPZ  = os.path.join(NPZ_DIR, f"profiles64.deduplicated.{TS}.npz")
OUT_INDEX_REMAPPED = os.path.join(NPZ_DIR, f"index_deduplicated.remapped.{TS}.csv")

# ===================== Helpers =====================
def parse_int(x):
    try: return int(str(x).strip())
    except: return None

def within_pct(x, y, pct):
    x=float(x); y=float(y)
    denom = max(abs(x), abs(y), 1e-12)
    return abs(x - y) <= (pct/100.0) * denom

ELNUM = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")
def parse_formula_sum(s):
    if not s: return {}
    s = str(s).strip().strip("'").strip('"')
    comp = defaultdict(float)
    for el, num in ELNUM.findall(s.replace(" ", "")):
        comp[el] += float(num) if num else 1.0
    return dict(comp)

def comp_key(formula_sum):
    comp = parse_formula_sum(formula_sum)
    if not comp: return None
    tot = sum(comp.values())
    if tot <= 0: return None
    norm = [(el, round(val/tot, 6)) for el,val in comp.items()]
    norm.sort(key=lambda x:x[0])
    return tuple(norm)

def parse_cif_fields(cif):
    if not cif: return {}
    def grab(tag, cast=float):
        m = re.search(rf"(?m)^{re.escape(tag)}\s+(.+)$", cif)
        if not m: return None
        raw = m.group(1).strip().strip("'").strip('"')
        try: return cast(raw)
        except:
            try: return float(raw)
            except: return raw
    def grab_int(tag):
        v = grab(tag, float)
        try: return int(round(float(v)))
        except: return None
    return {
        "sg":    grab_int("_symmetry_Int_Tables_number"),
        "a":     grab("_cell_length_a"),
        "b":     grab("_cell_length_b"),
        "c":     grab("_cell_length_c"),
        "alpha": grab("_cell_angle_alpha"),
        "beta":  grab("_cell_angle_beta"),
        "gamma": grab("_cell_angle_gamma"),
        "V":     grab("_cell_volume"),
        "formula": grab("_chemical_formula_sum", str),
    }

FIELDS = ("a","b","c","alpha","beta","gamma","V")

def id_priority(rid: str):
    rid = (rid or "").lower()
    if rid.startswith("mp-"):  return (0, rid)
    if rid.startswith("cod-") or rid.startswith("cod"): return (1, rid)
    return (2, rid)

def identical_pair_by_cif(r1, r2):
    if r1["cif_sg"] != r2["cif_sg"]: return False
    if r1["comp_key"] != r2["comp_key"]: return False
    # <-- CHANGED: use 3×PCT for V, PCT for others
    for k in FIELDS:
        tol = PCT_VOLUME if k == "V" else PCT
        if not within_pct(r1[k], r2[k], pct=tol): 
            return False
    return True

def cat_sg_equal(r1, r2):
    return (r1["cat_sg"] is not None and r2["cat_sg"] is not None and r1["cat_sg"] == r2["cat_sg"])

class DSU:
    def __init__(self): self.p = {}
    def find(self, x):
        if x not in self.p: self.p[x] = x
        if self.p[x] != x: self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)

def rel_dev(x, m):
    if m == 0: return 0.0
    return (x/m) - 1.0

# ===================== Load catalog & metadata =====================
print(f"[1/5] Loading catalog: {CATALOG}")
with open(CATALOG, "r", encoding="utf-8") as f:
    cat_rows = list(csv.DictReader(f))
print(f"  rows in catalog: {len(cat_rows):,}")

id_to_catrow = {r["id"]: r for r in cat_rows if r.get("id")}
cat_ids = list(id_to_catrow.keys())

print(f"[2/5] Loading metadata JSON: {METAJS}")
with open(METAJS, "r", encoding="utf-8") as f:
    meta = json.load(f)

# Build comparable records (CIF + catalog SG)
usable = []
skipped_no_meta = 0
skipped_missing_fields = 0

for rid in cat_ids:
    rec = meta.get(rid)
    if not rec:
        skipped_no_meta += 1
        continue
    cif = rec.get("cif_content","")
    fields = parse_cif_fields(cif)
    cif_sg = fields.get("sg")
    if cif_sg is None:
        try: cif_sg = int(rec.get("space_group"))
        except: pass
    ck = comp_key(fields.get("formula"))
    cat_sg = parse_int(id_to_catrow[rid].get("space_group"))
    if cif_sg is None or ck is None or any(fields.get(k) is None for k in FIELDS):
        skipped_missing_fields += 1
        continue
    usable.append({
        "id": rid,
        "cif_sg": cif_sg,
        "cat_sg": cat_sg,
        "comp_key": ck,
        "formula": fields.get("formula"),
        **{k: float(fields[k]) for k in FIELDS},
    })

print(f"  usable with CIF+catalog SG: {len(usable):,}  | skipped no meta: {skipped_no_meta:,}  | skipped missing fields: {skipped_missing_fields:,}")

# ===================== Blocking + clustering (with catalog SG safety) =====================
print("[3/5] Clustering duplicates (CIF rule + catalog SG safety)…")
buckets = defaultdict(list)
for r in usable:
    buckets[(r["cif_sg"], r["comp_key"])] .append(r)

dsu = DSU()
blocked_pairs = []  # (id1,id2,cif_sg,cat_sg1,cat_sg2,reason)
for key, group in buckets.items():
    if len(group) < 2: 
        continue
    group_sorted = sorted(group, key=lambda x: x["V"])
    n = len(group_sorted)
    for i in range(n):
        r1 = group_sorted[i]
        j = i + 1
        # <-- CHANGED: candidate window on V uses 3×PCT
        while j < n and within_pct(group_sorted[j]["V"], r1["V"], pct=PCT_VOLUME):
            r2 = group_sorted[j]
            if identical_pair_by_cif(r1, r2):
                if cat_sg_equal(r1, r2):
                    assert r1["cif_sg"] == r2["cif_sg"]
                    dsu.union(r1["id"], r2["id"])
                else:
                    reason = "catalog_sg_mismatch" if (r1["cat_sg"] is not None and r2["cat_sg"] is not None) else "catalog_sg_missing"
                    blocked_pairs.append((r1["id"], r2["id"], r1["cif_sg"], r1["cat_sg"], r2["cat_sg"], reason))
            j += 1

clusters = defaultdict(list)
for r in usable:
    root = dsu.find(r["id"])
    clusters[root].append(r)

dup_clusters = [g for g in clusters.values() if len(g) > 1]
dup_clusters.sort(key=lambda g: (-len(g), sorted(x["id"] for x in g)[0]))
print(f"  found {len(dup_clusters)} duplicate clusters (size>1)")

def cluster_is_identical_by_cif(group):
    for i in range(len(group)):
        for j in range(i+1, len(group)):
            if not identical_pair_by_cif(group[i], group[j]):
                return False
    return True

def choose_representative(group):
    if cluster_is_identical_by_cif(group):
        rep = sorted((r["id"] for r in group), key=id_priority)[0]
        return rep, "identical_mp_preferred"
    med = {k: statistics.median(r[k] for r in group) for k in FIELDS}
    best_id, best_score = None, float("inf")
    for r in group:
        score = sum((rel_dev(r[k], med[k])**2 for k in FIELDS))
        if score < best_score or (score == best_score and id_priority(r["id"]) < id_priority(best_id or r["id"])):
            best_score, best_id = score, r["id"]
    return best_id, "median_central"

# Compute drops/keeps + logs
rep_map = {}   # member -> representative
drop_ids = set()
keep_ids  = set()
sum_lines = []
sg_inconsistency_lines = []

with open(OUT_LOG, "w", newline="", encoding="utf-8") as logf:
    lw = csv.writer(logf)
    lw.writerow([
        "cluster_id","status","representative_id","member_id","keep",
        "cif_sg","catalog_sg","catalog_sg_consistent",
        "formula","a","b","c","alpha","beta","gamma","V",
        "dev_a_pct","dev_b_pct","dev_c_pct","dev_alpha_pct","dev_beta_pct","dev_gamma_pct","dev_V_pct",
        "reason"
    ])
    for group in dup_clusters:
        rep_id, reason = choose_representative(group)
        status = "identical" if reason == "identical_mp_preferred" else "similar"
        for r in group:
            rep_map[r["id"]] = rep_id
        for r in group:
            if r["id"] != rep_id: drop_ids.add(r["id"])
            else: keep_ids.add(r["id"])
        # summary & per-row logging
        vals = {k:[r[k] for r in group] for k in FIELDS}
        rng_pct = {k: ( (max(vals[k]) - min(vals[k])) / max(max(vals[k]),1e-12) * 100.0 ) for k in FIELDS}
        cif_sg = group[0]["cif_sg"]
        formula = group[0]["formula"]
        members = sorted(r["id"] for r in group)
        removed = [m for m in members if m != rep_id]
        cat_sg_set = sorted({r["cat_sg"] for r in group if r["cat_sg"] is not None})
        cat_sg_unique = int(len(cat_sg_set) == 1)

        print(f"  [{status.upper():8}] CIF SG #{cif_sg:>3}  n={len(group)}  rep={rep_id}  removed={removed}  "
              f"catalog_SG_unique={bool(cat_sg_unique)} cat_SGs={cat_sg_set or ['None']}")

        sum_lines.append(
            f"[{status.upper()}] CIF SG #{cif_sg} | formula='{formula}' | size={len(group)} | rep={rep_id} | removed={removed} | "
            + " | ".join([f"{k}: range={rng_pct[k]:.3f}%" for k in FIELDS])
            + f" | catalog_SG_unique={bool(cat_sg_unique)} cat_SGs={cat_sg_set or ['None']}"
        )

        if not cat_sg_unique:
            sg_inconsistency_lines.append(
                f"Cluster (CIF SG #{cif_sg}, comp={formula}): mixed catalog SGs -> {cat_sg_set or ['None']} | ids={members}"
            )

        med = {k: statistics.median(vals[k]) for k in FIELDS}
        for r in group:
            lw.writerow([
                rep_id, status, rep_id, r["id"], int(r["id"]==rep_id),
                r["cif_sg"], r["cat_sg"], int(r["cat_sg"] == group[0]["cat_sg"] if r["cat_sg"] is not None and group[0]["cat_sg"] is not None else 0),
                formula, f"{r['a']:.6f}", f"{r['b']:.6f}", f"{r['c']:.6f}",
                f"{r['alpha']:.6f}", f"{r['beta']:.6f}", f"{r['gamma']:.6f}", f"{r['V']:.6f}",
                f"{abs(rel_dev(r['a'],med['a']))*100:.6f}",
                f"{abs(rel_dev(r['b'],med['b']))*100:.6f}",
                f"{abs(rel_dev(r['c'],med['c']))*100:.6f}",
                f"{abs(rel_dev(r['alpha'],med['alpha']))*100:.6f}",
                f"{abs(rel_dev(r['beta'],med['beta']))*100:.6f}",
                f"{abs(rel_dev(r['gamma'],med['gamma']))*100:.6f}",
                f"{abs(rel_dev(r['V'],med['V']))*100:.6f}",
                reason
            ])

# ===================== Write deduped catalog =====================
kept_rows = [r for r in cat_rows if r.get("id") not in drop_ids]
with open(OUT_CAT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cat_rows[0].keys() if cat_rows else [])
    w.writeheader()
    for r in kept_rows:
        w.writerow(r)

# ===================== Update profiles index (drop dup IDs) =====================
print(f"[4/5] Updating profiles index & NPZ…")
if not os.path.exists(INDEX):
    print("  WARNING: profiles index not found; skipping index & NPZ steps.")
    idx_rows = []
else:
    with open(INDEX, "r", encoding="utf-8") as f:
        idx_rows = list(csv.DictReader(f))
    print(f"  index rows: {len(idx_rows):,}")

# Drop duplicates from index; keep unknown IDs
index_kept_rows = [r for r in idx_rows if r.get("id") and r["id"] not in drop_ids]

# Write index_deduplicated.csv (keeps original row numbers)
if idx_rows:
    with open(OUT_INDEX_DROPSAFE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=idx_rows[0].keys())
        w.writeheader()
        for r in index_kept_rows:
            w.writerow(r)

# ===================== Filter NPZ -> new NPZ + remapped index =====================
made_npz = False
if idx_rows and os.path.exists(NPZ):
    # Select rows by original row order (sorted)
    try:
        kept_sorted = sorted(index_kept_rows, key=lambda r: int(r["row"]))
    except Exception as e:
        raise SystemExit(f"Index 'row' must be integer-like. Problem: {e}")
    row_ids = [int(r["row"]) for r in kept_sorted]
    ids_in_order = [r["id"] for r in kept_sorted]

    npz = np.load(NPZ, allow_pickle=False)
    profiles   = npz["profiles"]
    q_min      = npz["q_min"]; q_max = npz["q_max"]
    n_bins     = npz["n_bins"]; sigma_bins = npz["sigma_bins"]

    if row_ids and max(row_ids) >= profiles.shape[0]:
        raise SystemExit(f"Some index rows exceed profiles rows: max row {max(row_ids)} vs profiles {profiles.shape[0]}")

    new_profiles = profiles[row_ids, :] if row_ids else profiles[:0, :]
    np.savez_compressed(
        OUT_NPZ,
        profiles=new_profiles.astype(profiles.dtype, copy=False),
        q_min=q_min, q_max=q_max, n_bins=n_bins, sigma_bins=sigma_bins
    )
    made_npz = True

    # Remapped index for the NEW NPZ
    with open(OUT_INDEX_REMAPPED, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id","row"])
        for new_row, _id in enumerate(ids_in_order):
            w.writerow([_id, new_row])

# ===================== Summary file =====================
total_cat = len(cat_rows)
kept_cat  = len(kept_rows)
removed_cat = total_cat - kept_cat

total_idx = len(idx_rows)
kept_idx  = len(index_kept_rows)
removed_idx = total_idx - kept_idx

with open(OUT_SUM, "w", encoding="utf-8") as sf:
    sf.write(f"De-duplication run @ {TS}\n")
    sf.write(f"Catalog: {CATALOG}\nMeta:    {METAJS}\nIndex:   {INDEX}\nNPZ:     {NPZ}\n")
    # <-- CHANGED: make tolerances explicit
    sf.write(f"Tolerance: ±{PCT}% on (a,b,c,alpha,beta,gamma) and ±{PCT_VOLUME}% on V\n\n")
    sf.write(f"[Catalog]\n")
    sf.write(f"  Rows in catalog: {total_cat:,}\n")
    sf.write(f"  Usable with CIF+catalog SG: {len(usable):,} (skipped no meta: {skipped_no_meta:,}; missing fields: {skipped_missing_fields:,})\n")
    sf.write(f"  Duplicate clusters (post safety check): {len(dup_clusters):,}\n")
    sf.write(f"  Removed rows: {removed_cat:,}\n")
    sf.write(f"  Kept rows:    {kept_cat:,}\n\n")
    sf.write("  Clusters:\n")
    for line in sum_lines:
        sf.write("  " + line + "\n")
    sf.write("\n  Catalog/CIF SG inconsistencies (CIF says same SG; catalog SG differs or missing blocks merging):\n")
    if blocked_pairs:
        for (i1,i2,sg,c1,c2,reason) in blocked_pairs:
            sf.write(f"   - {i1} vs {i2} | CIF SG #{sg} | catalog SGs: {c1} vs {c2} | {reason}\n")
    else:
        sf.write("   - None\n")

    sf.write(f"\n[Profiles index]\n")
    if idx_rows:
        sf.write(f"  Input rows:   {total_idx:,}\n")
        sf.write(f"  Removed rows: {removed_idx:,}\n")
        sf.write(f"  Kept rows:    {kept_idx:,}\n")
        sf.write(f"  Output (drop-only, original rows): {OUT_INDEX_DROPSAFE}\n")
    else:
        sf.write("  (No index file processed)\n")

    sf.write(f"\n[NPZ]\n")
    if made_npz:
        sf.write(f"  Input profiles shape: {profiles.shape}\n")
        sf.write(f"  Output profiles shape: {new_profiles.shape}\n")
        sf.write(f"  New NPZ: {OUT_NPZ}\n")
        sf.write(f"  New remapped index: {OUT_INDEX_REMAPPED}\n")
    else:
        sf.write("  (No NPZ written — missing index/npz file or zero kept rows)\n")

# ===================== Console summary =====================
print("\n----- DONE -----")
print(f"Deduped catalog:   {OUT_CAT}")
print(f"Log (CSV):         {OUT_LOG}")
print(f"Summary (TXT):     {OUT_SUM}")
if 'idx_rows' in locals() and idx_rows:
    print(f"Index (drop-only): {OUT_INDEX_DROPSAFE}")
if 'made_npz' in locals() and made_npz:
    print(f"New NPZ:           {OUT_NPZ}")
    print(f"Remapped index:    {OUT_INDEX_REMAPPED}")
print(f"Catalog removed {removed_cat:,} / kept {kept_cat:,}")
if 'idx_rows' in locals() and idx_rows:
    print(f"Index   removed {removed_idx:,} / kept {kept_idx:,}")

