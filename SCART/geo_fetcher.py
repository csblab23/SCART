import GEOparse
import os
import tarfile
import scanpy as sc
import anndata as ad
import pandas as pd
import gzip
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

TABULA_DOI_LINK = "https://doi.org/10.6084/m9.figshare.27921984"

TABULA_FILES = {
    "bladder_cancer":       "Bladder_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "blood_cancer":         "Blood_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "bone_marrow_cancer":   "Bone_Marrow_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "ear_cancer":           "Ear_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "eye_cancer":           "Eye_TSP1_30_version2d_10X_smartseq_scvi_Nov122024_updated.h5ad",
    "fat_cancer":           "Fat_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "heart_cancer":         "Heart_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "kidney_cancer":        "Kidney_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "large_intestine_cancer": "Large_Intestine_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "liver_cancer":         "Liver_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "lung_cancer":          "Lung_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "lymph_node_cancer":    "Lymph_Node_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "breast_cancer":        "Mammary_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "muscle_cancer":        "Muscle_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "ovary_cancer":         "Ovary_TSP1_30_version2d_10X_smartseq_scvi_Nov262024.h5ad",
    "pancreas_cancer":      "Pancreas_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "prostate_cancer":      "Prostate_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "salivary_gland_cancer": "Salivary_Gland_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "skin_cancer":          "Skin_TSP1_30_version2d_10X_smartseq_scvi_Nov122024_updated.h5ad",
    "small_intestine_cancer": "Small_Intestine_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "spleen_cancer":        "Spleen_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "stomach_cancer":       "Stomach_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "testis_cancer":        "Testis_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "thymus_cancer":        "Thymus_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "tongue_cancer":        "Tongue_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "trachea_cancer":       "Trachea_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "uterus_cancer":        "Uterus_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "vasculature_cancer":   "Vasculature_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
}

VALID_CANCER_TYPES = sorted(TABULA_FILES.keys())

DISEASE_TUMOR_KEYWORDS = [
    "tumor","tumour","cancer","carcinoma","adenocarcinoma","malignant","malignancy",
    "metastatic","metastasis","neoplasm","neoplastic","CAR-T","infusion",
    "pre-infusion","post-infusion","leukapheresis","leukemia","leukaemia","lymphoma",
    "myeloma","aml","cml","all","cll","mds","acute myeloid","chronic myeloid",
    "acute lymphoblastic","chronic lymphocytic","acute lymphocytic","t-cell leukemia",
    "b-cell leukemia","hairy cell leukemia","large granular lymphocyte",
    "myelodysplastic","myeloproliferative","polycythemia vera","essential thrombocythemia",
    "myelofibrosis","dlbcl","follicular lymphoma","mantle cell lymphoma","burkitt lymphoma",
    "hodgkin","non-hodgkin","marginal zone lymphoma","anaplastic large cell",
    "primary mediastinal b-cell","multiple myeloma","plasma cell dyscrasia","plasmacytoma",
    "waldenström","smoldering myeloma","amyloidosis","hgsoc","lgsoc","pdac","nsclc","sclc",
    "gbm","glioblastoma","glioma","astrocytoma","melanoma","sarcoma","blastoma",
    "hepatocellular","cholangiocarcinoma","seminoma","teratoma","tnbc","triple negative",
    "her2+","her2-positive","her2 positive","er+","er-positive","er positive",
    "pr+","pr-positive","pr positive","luminal a","luminal b","dcis","invasive ductal",
    "invasive lobular","t1n","t2n","t3n","t4n","tnm stage","relapsed","refractory",
    "recurrent","post-treatment","post treatment","residual disease",
    "pathologic complete response","overall survival","disease-free survival",
]


# ── Module-level helpers ──────────────────────────────────────────────────────

def _read_10x_h5_via_h5py(file_path: str):
    import h5py, scipy.sparse as sp

    def _decode(arr):
        return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr]

    try:
        with h5py.File(file_path, "r") as f:
            if "matrix" in f:
                g = f["matrix"]
                data, indices, indptr = g["data"][:], g["indices"][:], g["indptr"][:]
                shape    = tuple(g["shape"][:])
                barcodes = _decode(g["barcodes"][:])
                feat     = g["features"]
                gene_ids   = _decode(feat["id"][:])
                gene_names = _decode(feat["name"][:]) if "name" in feat else gene_ids
            else:
                genome_key = next(
                    (k for k in f.keys() if isinstance(f[k], h5py.Group) and "data" in f[k]),
                    None
                )
                if genome_key is None:
                    return None
                g = f[genome_key]
                data, indices, indptr = g["data"][:], g["indices"][:], g["indptr"][:]
                shape      = tuple(g["shape"][:])
                barcodes   = _decode(g["barcodes"][:])
                gene_ids   = _decode(g["gene_ids"][:])   if "gene_ids"   in g else _decode(g["gene_names"][:])
                gene_names = _decode(g["gene_names"][:]) if "gene_names" in g else gene_ids

        X   = sp.csr_matrix((data, indices, indptr), shape=shape).T
        obs = pd.DataFrame(index=barcodes)
        var = pd.DataFrame({"gene_ids": gene_ids, "gene_symbols": gene_names}, index=gene_names)
        return ad.AnnData(X=X, obs=obs, var=var)
    except Exception as exc:
        print(f"    h5py fallback read failed: {exc}")
        return None


def _dedup_var_names(adata: ad.AnnData) -> ad.AnnData:
    if adata.var_names.is_unique:
        return adata
    seen, new_idx = {}, []
    for v in adata.var_names:
        if v in seen:
            seen[v] += 1
            new_idx.append(f"{v}.{seen[v]}")
        else:
            seen[v] = 0
            new_idx.append(v)
    adata.var_names = new_idx
    return adata


def _safe_concat(adatas: list) -> ad.AnnData:
    adatas   = [_dedup_var_names(a) for a in adatas]
    all_names = []
    for a in adatas:
        all_names.extend(a.var_names.tolist())
    union = list(dict.fromkeys(all_names))

    if len(set(union)) == len(union):
        return ad.concat(adatas, join="outer")

    print("  Post-dedup union var_names still contain duplicates; applying global remapping …")
    seen, global_map = {}, {}
    for name in all_names:
        if name not in global_map:
            if name not in seen:
                seen[name] = 0
                global_map[name] = name
            else:
                seen[name] += 1
                global_map[name] = f"{name}.{seen[name]}"

    remapped = []
    for a in adatas:
        a = a.copy()
        a.var_names = [global_map.get(v, v) for v in a.var_names]
        remapped.append(a)
    return ad.concat(remapped, join="outer")


# ── SampleAnnotator ───────────────────────────────────────────────────────────

class SampleAnnotator:
    def __init__(self, *inputs, cancer_type: str, min_genes=None,
                 max_mt=None, manual_annotation_col=None, batch_key=None):
        self.inputs   = list(inputs)
        self.base_dir = "GSE_data"
        self.min_genes = min_genes
        self.max_mt    = max_mt

        if not cancer_type or not isinstance(cancer_type, str):
            raise ValueError(
                "\ncancer_type is required.\n"
                "  cancer_type='blood_cancer'\n"
                "To see all keys:\n"
                "  from SCART.geo_fetcher import VALID_CANCER_TYPES"
            )

        self._user_cancer_type, self._tabula_types, self._unknown_types = (
            self._parse_cancer_type(cancer_type)
        )
        self.manual_annotation_col = manual_annotation_col

        # CLAUDE EDIT — optional keyword to search for in GEO sample metadata
        # (e.g. "donor", "patient", "timepoint") to build a batch column for
        # GSE-derived data. Only meaningful for GEO ID inputs — has no effect
        # on user-supplied h5ad inputs, since those already carry their own
        # obs columns untouched. If not set, 'gsm_id' remains the batch
        # column downstream (Module 2 picks it up automatically).
        self.batch_key = batch_key.strip() if isinstance(batch_key, str) else batch_key

        os.makedirs(self.base_dir, exist_ok=True)

        self.gse_ids     = []
        self.h5ad_inputs = []
        for item in self.inputs:
            if isinstance(item, str) and item.lower().endswith(".h5ad"):
                self.h5ad_inputs.append(item)
            else:
                self.gse_ids.append(item)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _parse_cancer_type(self, cancer_type):
        tokens        = [t.strip() for t in cancer_type.split(",") if t.strip()]
        tabula_types  = [t for t in tokens if t in TABULA_FILES]
        unknown_types = [t for t in tokens if t not in TABULA_FILES]
        return ", ".join(tokens), tabula_types, unknown_types

    def _store_qc_params(self, adata):
        if self.min_genes is None and self.max_mt is None:
            adata.uns.pop("qc_params", None)
            return
        adata.uns["qc_params"] = {"min_genes": self.min_genes, "max_mt": self.max_mt}

    def _apply_qc(self, adata):
        """
        Apply QC filtering using min_genes and max_mt if provided.
        - min_genes  : minimum number of genes expressed per cell
        - max_mt     : maximum mitochondrial gene % per cell
        Modifies adata in-place and returns the filtered AnnData.
        """
        if self.min_genes is None and self.max_mt is None:
            return adata

        n_cells_before = adata.n_obs
        print(f"\n========== QC Filtering ==========")
        print(f"  Cells before QC : {n_cells_before}")

        # Compute basic QC metrics (n_genes_by_counts, total_counts, pct_counts_mt)
        adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
        )

        mask = pd.Series([True] * adata.n_obs, index=adata.obs_names)

        if self.min_genes is not None:
            gene_mask = adata.obs["n_genes_by_counts"] >= self.min_genes
            n_fail    = (~gene_mask).sum()
            print(f"  Cells failing min_genes ({self.min_genes}) : {n_fail}")
            mask = mask & gene_mask

        if self.max_mt is not None:
            mt_mask = adata.obs["pct_counts_mt"] <= self.max_mt
            n_fail  = (~mt_mask).sum()
            print(f"  Cells failing max_mt ({self.max_mt}%)       : {n_fail}")
            mask = mask & mt_mask

        adata = adata[mask].copy()
        print(f"  Cells after QC  : {adata.n_obs}  (removed {n_cells_before - adata.n_obs})")
        return adata

    def _store_manual_annotation(self, adata, source_file):
        col = self.manual_annotation_col
        if col not in adata.obs.columns:
            raise ValueError(
                f"\nmanual_annotation_col='{col}' not found in adata.obs of '{source_file}'.\n"
                f"Available: {list(adata.obs.columns)}"
            )
        if "popv_majority_vote_prediction" in adata.obs.columns:
            print(f"  WARNING: overwriting 'popv_majority_vote_prediction' with '{col}'.")
        adata.obs["popv_majority_vote_prediction"] = adata.obs[col].astype(str)
        adata.uns["manual_annotation_col"] = col
        adata.uns["skip_popv"]             = True

        unique_labels  = sorted(adata.obs["popv_majority_vote_prediction"].unique())
        epithelial     = [l for l in unique_labels if "epithelial cell" in l.lower()]
        non_epithelial = [l for l in unique_labels if "epithelial cell" not in l.lower()]
        print(f"\n  Manual annotation column : '{col}'")
        print(f"  Copied to               : 'popv_majority_vote_prediction'")
        print(f"  Total unique labels     : {len(unique_labels)}")
        print(f"  Epithelial labels found : {epithelial if epithelial else 'NONE'}")
        print(f"  Non-epithelial labels   : {non_epithelial}")
        print( "  PopV will be SKIPPED (adata.uns['skip_popv'] = True)")
        if not epithelial:
            print(
                "\n  ⚠ WARNING: No epithelial labels detected.\n"
                "  Labels must END WITH 'epithelial cell' (case-insensitive)."
            )

    # CLAUDE EDIT — search a GSM's GEO metadata for a field matching
    # self.batch_key (case-insensitive substring on the field's key),
    # GEO characteristics are typically formatted as "key: value" strings
    # (e.g. "donor: P1", "patient id: 3"). Returns the value, or None if
    # no matching field was found anywhere in this GSM's metadata.
    def _extract_batch_hint_value(self, gsm):
        hint = self.batch_key.lower()

        # Priority fields first (same fields _classify_gsm already trusts)
        priority_fields = []
        for key in ("characteristics_ch1", "title", "source_name_ch1"):
            priority_fields.extend(gsm.metadata.get(key, []))

        for field in priority_fields:
            if isinstance(field, str) and ":" in field:
                k, _, v = field.partition(":")
                if hint in k.strip().lower():
                    return v.strip()

        # Fallback — scan every metadata field for a "key: value" match
        for key, values in gsm.metadata.items():
            for field in values:
                if isinstance(field, str) and ":" in field:
                    k, _, v = field.partition(":")
                    if hint in k.strip().lower():
                        return v.strip()

        return None

    def _print_reference_guidance(self):
        print("\n========== REFERENCE GUIDANCE ==========")
        print(f"Cancer type(s) provided: {self._user_cancer_type}\n")
        if self.h5ad_inputs:
            if self.manual_annotation_col:
                print("👉 h5ad provided WITH manual annotations — PopV SKIPPED.")
            else:
                print("👉 h5ad provided — supply your own reference for PopV.")
        for ct in self._tabula_types:
            print(f"\n✅ Tabula Sapiens reference available: {ct}")
            print(f"   Download : {TABULA_FILES[ct]}")
            print(f"   From     : {TABULA_DOI_LINK}")
        for ct in self._unknown_types:
            print(f"\n⚠️  '{ct}' not in Tabula Sapiens — supply your own reference.")

    # CLAUDE EDIT — explains what batch key will be used downstream, and
    # what to do if the default (GSM ID) isn't right for this dataset.
    def _print_batch_guidance(self):
        print("\n========== BATCH KEY GUIDANCE ==========")
        if self.h5ad_inputs and not self.gse_ids:
            print("Input was h5ad file(s) — batch_key does not apply (GEO-only feature).")
            print("SCART does not know what you named your batch/donor column, so it")
            print("cannot detect or set one for you here. If your h5ad already has its")
            print("own batch/donor column, pass its EXACT name as")
            print("batch_key='<column name>' to Module 2 (popv_annotation.auto_run_popv).")
            return

        if self.h5ad_inputs and self.gse_ids:
            print("Input mixes GEO ID(s) with your own h5ad file(s).")
            print("batch_key (if set) only extracts a value for the GEO-derived cells —")
            print("your own h5ad's cells keep whatever columns they already had,")
            print("completely untouched (SCART does not know what you named them).")
            if self.batch_key:
                print(f"  You set batch_key='{self.batch_key}': GEO-derived cells get this")
                print(f"  column extracted from GEO metadata where a match is found.")
                print(f"  For ONE unified batch column across both sources, your own")
                print(f"  h5ad must ALSO already contain a column named exactly")
                print(f"  '{self.batch_key}' — otherwise those rows will be missing a")
                print(f"  value for that column once Module 1 combines everything.")
            else:
                print("  You did not set batch_key: GEO-derived cells fall back to")
                print("  'gsm_id'/'gse_id'; your own h5ad's cells have neither column,")
                print("  so those rows will have no value there after combining.")
                print("  For a clean, single batch column across both sources:")
                print("    1. Give your own h5ad a batch/donor column, then pass that")
                print("       SAME name as batch_key='<name>' to BOTH this SampleAnnotator")
                print("       call (so the GEO cells get it extracted from metadata) and")
                print("       to Module 2, or")
                print("    2. Add a batch/donor column to the combined h5ad yourself")
                print("       after Module 1 runs, then pass its name as batch_key= to")
                print("       Module 2 directly.")
            return

        if self.batch_key:
            print(f"batch_key='{self.batch_key}' was requested for GEO-derived data.")
            print(f"  Where a match was found, it was saved as adata.obs['{self.batch_key}'].")
            print(f"  Where no match was found, that sample falls back to 'unknown' for this column.")
            print(f"  → Pass batch_key='{self.batch_key}' to Module 2 to use it.")
        else:
            print("No batch_key provided — Module 2 will default to using")
            print("GSM ID ('gsm_id', one value per sample) as the batch key.")
            print("This is usually correct (one GSM = one library/capture run).")
            print()
            print("If GSM ID is NOT a valid batch for your data — e.g. one GSM actually")
            print("contains multiple pooled patients/donors — you have two options:")
            print("  1. Re-run Module 1 with batch_key='<keyword>' (e.g. 'donor',")
            print("     'patient') to try extracting it automatically from GEO sample")
            print("     metadata (characteristics_ch1 / title / source_name_ch1).")
            print("  2. Supply your own h5ad with a batch column already in adata.obs,")
            print("     then pass batch_key='<your column name>' to Module 2.")

    # ── Public entry-point ────────────────────────────────────────────────

    def run(self):
        normal = []; tumor = []; unspecified = []; annotation_info = {}
        tumor_adatas = []; results = {}

        for gse_id in self.gse_ids:
            n, t, u, ann, batch_hint_map = self._process_gse(gse_id)
            normal.extend(n); tumor.extend(t); unspecified.extend(u)
            annotation_info.update(ann)

            adata = self._build_h5ad(
                gse_id, t,
                save_single=(len(self.gse_ids) == 1 and len(self.h5ad_inputs) == 0),
                batch_hint_map=batch_hint_map,
            )
            if adata is not None:
                tumor_adatas.append(adata)
            results[gse_id] = (n, t, u, ann, None, self._user_cancer_type)

        for file in self.h5ad_inputs:
            print("\n========== Reading h5ad file ==========")
            adata = sc.read_h5ad(file)
            adata.obs_names_make_unique()
            adata.layers["counts"] = adata.X.copy()
            adata.raw = adata
            adata.uns["cancer_type"] = self._user_cancer_type
            print(f"  cancer_type stored: {self._user_cancer_type}")
            if self.manual_annotation_col is not None:
                print("\n  Manual annotation mode activated.")
                self._store_manual_annotation(adata, file)
            self._store_qc_params(adata)
            adata = self._apply_qc(adata)
            tumor_adatas.append(adata)
            results[file] = ([], [], [], {}, None, self._user_cancer_type)

        query_h5ad   = None
        total_inputs = len(self.gse_ids) + len(self.h5ad_inputs)

        if total_inputs == 1:
            if len(self.gse_ids) == 1:
                query_h5ad = f"{self.gse_ids[0]}_tumor.h5ad"
                results[self.gse_ids[0]] = (
                    normal, tumor, unspecified, annotation_info,
                    query_h5ad, self._user_cancer_type
                )
            elif len(self.h5ad_inputs) == 1:
                adata    = tumor_adatas[0]
                filename = "input_tumor.h5ad"
                adata.write(filename)
                print("\n========== h5ad created ==========")
                print(f"{filename} is created successfully")
                if self.manual_annotation_col:
                    print(f"Manual annotation stored → col='{self.manual_annotation_col}', skip_popv=True")
                    print("Next step: run Module 3 (preprocessing) directly.")
                if self.min_genes is not None or self.max_mt is not None:
                    print(f"QC applied and params stored → min_genes={self.min_genes}, max_mt={self.max_mt}")
                else:
                    print("QC step disabled (no min_genes / max_mt — skipped in Module 3)")
                query_h5ad = filename
                results[self.h5ad_inputs[0]] = ([], [], [], {}, query_h5ad, self._user_cancer_type)

        elif total_inputs > 1 and tumor_adatas:
            combined = ad.concat(tumor_adatas, join="outer")
            combined.obs_names_make_unique()
            combined.layers["counts"] = combined.X.copy()
            combined.raw = combined
            combined.uns["cancer_type"] = self._user_cancer_type
            print(f"\n  cancer_type stored in combined h5ad: {self._user_cancer_type}")

            if self.manual_annotation_col is not None:
                if "popv_majority_vote_prediction" in combined.obs.columns:
                    combined.uns["manual_annotation_col"] = self.manual_annotation_col
                    combined.uns["skip_popv"]             = True
                    print(f"\n  Combined h5ad: manual annotation carried through from '{self.manual_annotation_col}'.")
                else:
                    print("\n  WARNING: 'popv_majority_vote_prediction' lost during concat.")

            self._store_qc_params(combined)
            combined = self._apply_qc(combined)
            combined.write("combined_tumor.h5ad")
            print("\n========== h5ad created ==========")
            print("combined_tumor.h5ad is created successfully")
            if self.manual_annotation_col:
                print(f"Manual annotation stored → col='{self.manual_annotation_col}', skip_popv=True")
            if self.min_genes is not None or self.max_mt is not None:
                print(f"QC applied and params stored → min_genes={self.min_genes}, max_mt={self.max_mt}")
            else:
                print("QC step disabled (no min_genes / max_mt — skipped in Module 3)")
            query_h5ad = "combined_tumor.h5ad"
            for key in results:
                n, t, u, ann, _, ct = results[key]
                results[key]        = (n, t, u, ann, query_h5ad, ct)

        self._print_reference_guidance()
        self._print_batch_guidance()
        return (normal, tumor, unspecified, annotation_info,
                query_h5ad, self._user_cancer_type, results)

    # ── GEO processing ────────────────────────────────────────────────────

    def _classify_gsm(self, gsm):
        normal_keywords = [
            "normal","healthy","control","adjacent normal",
            "non-tumor","non-tumour","non-cancer","benign","non-malignant",
        ]
        per_sample_fields = ["title", "source_name_ch1", "characteristics_ch1"]
        ps_text = " ".join(
            " ".join(gsm.metadata.get(f, [])) for f in per_sample_fields
        ).lower()

        has_normal_ps  = any(k in ps_text for k in normal_keywords)
        has_disease_ps = any(k in ps_text for k in DISEASE_TUMOR_KEYWORDS)
        if has_normal_ps and not has_disease_ps:
            return "normal"

        full_text = " ".join(str(v) for v in gsm.metadata.values()).lower()
        if any(k in full_text for k in DISEASE_TUMOR_KEYWORDS):
            return "tumor"
        if any(k in full_text for k in normal_keywords):
            return "normal"
        return "unspecified"

    def _download_gse_level_suppl(self, gse_id, gse_dir):
        import urllib.request, re
        series_stub = "GSE" + str(int(gse_id[3:]) // 1000) + "nnn"
        ftp_base    = (f"https://ftp.ncbi.nlm.nih.gov/geo/series/"
                       f"{series_stub}/{gse_id}/suppl/")
        try:
            with urllib.request.urlopen(ftp_base, timeout=30) as resp:
                listing = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            print(f"  Warning: could not fetch GSE-level FTP listing: {exc}")
            return

        pattern  = re.compile(r'href="(' + re.escape(gse_id) + r'[^"]+)"', re.IGNORECASE)
        gse_files = pattern.findall(listing)

        def _is_shared_ref(fname):
            fl = fname.lower()
            if fl.endswith("_raw.tar") or fl.endswith(".tar") or fl.endswith(".tar.gz"):
                return False
            return any(kw in fl for kw in (
                "features", "genes", "barcodes", "cell_types", "metadata",
                "count", "umi", "expression", "matrix", "cellinfo", "cell_info", "anno",
            ))

        for fname in gse_files:
            if not _is_shared_ref(fname):
                continue
            dest = os.path.join(gse_dir, fname)
            if os.path.exists(dest):
                continue
            print(f"  Downloading GSE-level supplementary file: {fname}")
            try:
                urllib.request.urlretrieve(ftp_base + fname, dest)
            except Exception as exc:
                print(f"  Warning: failed to download {fname}: {exc}")

    def _download_missing_gsm_suppl(self, gse, gse_dir):
        import urllib.request
        for gsm_id, gsm in gse.gsms.items():
            urls = []
            for key, val in gsm.metadata.items():
                if key.startswith("supplementary_file") and val:
                    urls.extend(val)
            if not urls:
                continue
            supp_dirs = [
                d for d in os.listdir(gse_dir)
                if d.startswith(f"Supp_{gsm_id}")
                   and os.path.isdir(os.path.join(gse_dir, d))
            ]
            if not supp_dirs:
                continue
            gsm_supp_dir = os.path.join(gse_dir, supp_dirs[0])
            for url in urls:
                url = url.strip()
                if not url or url == "NONE":
                    continue
                fname = url.split("/")[-1]
                dest  = os.path.join(gsm_supp_dir, fname)
                if os.path.exists(dest):
                    continue
                print(f"  Downloading missing supplementary file: {gsm_id}/{fname}")
                try:
                    urllib.request.urlretrieve(url, dest)
                except Exception as exc:
                    print(f"  Warning: failed to download {fname} for {gsm_id}: {exc}")

    def _process_gse(self, gse_id):
        gse_dir = os.path.join(self.base_dir, gse_id)
        os.makedirs(gse_dir, exist_ok=True)
        gse = GEOparse.get_GEO(geo=gse_id, destdir=gse_dir)
        self._download_gse_level_suppl(gse_id, gse_dir)

        def _supp_present(gsm_id):
            if os.path.isdir(os.path.join(gse_dir, gsm_id)):
                return True
            return any(
                d.startswith(f"Supp_{gsm_id}")
                for d in os.listdir(gse_dir)
                if os.path.isdir(os.path.join(gse_dir, d))
            )

        if all(_supp_present(gsm_id) for gsm_id in gse.gsms):
            print(f"  Supplementary files already present for {gse_id} — skipping download.")
        else:
            gse.download_supplementary_files(gse_dir)

        self._download_missing_gsm_suppl(gse, gse_dir)

        normal = []; tumor = []; unspecified = []
        annotation_info = {}
        batch_hint_map = {}
        excluded_non_scrna = []; excluded_non_human = []

        _SCRNA_KEYWORDS = ["rna-seq","scrna","single cell","single-cell",
                           "singlenucleus","single nucleus","snrna"]
        _SCRNA_EVIDENCE = [
            "feature_bc_matrix","filtered_feature","raw_feature","gene_bc_matrices",
            "barcodes.tsv","matrix.mtx","10x chromium","10x genomics",
            "chromium controller","dropseq","drop-seq","indrop","indrops",
            "scrna-seq","scrna seq","sc rna-seq","single-cell rna","single cell rna",
            "snrna-seq","snrna seq","single-nucleus rna","cellranger","cell ranger","seurat",
        ]

        for gsm_id, gsm in gse.gsms.items():
            organism = " ".join(gsm.metadata.get("organism_ch1", [])).lower()
            if "homo sapiens" not in organism:
                excluded_non_human.append(gsm_id)
                continue
            library  = " ".join(gsm.metadata.get("library_strategy", [])).lower()
            pass_a   = any(k in library for k in _SCRNA_KEYWORDS)
            if not pass_a:
                full_meta = " ".join(str(v) for v in gsm.metadata.values()).lower()
                if not any(k in full_meta for k in _SCRNA_EVIDENCE):
                    excluded_non_scrna.append(gsm_id)
                    continue

            label = self._classify_gsm(gsm)
            if label == "normal":
                normal.append(gsm_id)
            elif label == "tumor":
                tumor.append(gsm_id)
            else:
                unspecified.append(gsm_id)
            annotation_info[gsm_id] = label

            # CLAUDE EDIT — try to extract the requested batch_key
            # value from this GSM's own metadata text.
            if self.batch_key:
                batch_hint_map[gsm_id] = self._extract_batch_hint_value(gsm)

        print(f"\n========== SAMPLE SUMMARY: {gse_id} ==========")
        print(f"Cancer type (user-supplied): {self._user_cancer_type}")
        print("Normal samples:",       ", ".join(normal)             or "None")
        print("Tumor samples:",        ", ".join(tumor)              or "None")
        print("Unspecified samples:",  ", ".join(unspecified)        or "None")
        print("Excluded (non-human):", ", ".join(excluded_non_human) or "None")
        print("Excluded (non-scRNA):", ", ".join(excluded_non_scrna) or "None")

        if self.batch_key:
            found   = [g for g, v in batch_hint_map.items() if v is not None]
            missing = [g for g, v in batch_hint_map.items() if v is None]
            print(f"batch_key '{self.batch_key}': "
                  f"found in {len(found)}/{len(batch_hint_map)} samples")
            if missing:
                print(f"  WARNING: no '{self.batch_key}' field found for: "
                      f"{', '.join(missing)} — these will be tagged 'unknown'.")

        return normal, tumor, unspecified, annotation_info, batch_hint_map

    # ── Matrix readers ────────────────────────────────────────────────────

    def _read_generic_matrix(self, file_path):
        import scipy.sparse as sp
        fl = file_path.lower()
        if any(fl.endswith(e) for e in (".mtx", ".mtx.gz", ".h5", ".hdf5", ".h5.gz")):
            return None
        try:
            opener = gzip.open(file_path, "rt") if file_path.endswith(".gz") else open(file_path, "r")
            with opener as f:
                first_line = f.readline()
            sep = "\t" if "\t" in first_line else ","

            if file_path.endswith(".gz"):
                with gzip.open(file_path, "rt") as f:
                    df = pd.read_csv(f, sep=sep, index_col=0)
            else:
                df = pd.read_csv(file_path, sep=sep, index_col=0)

            if df.empty:
                return None

            def _index_is_strings(idx):
                sample = list(idx[:20])
                numeric = sum(1 for v in sample if _try_float(v))
                return numeric < len(sample) / 2

            def _try_float(v):
                try: float(v); return True
                except: return False

            idx_str = _index_is_strings(df.index)
            col_str = _index_is_strings(df.columns)

            if idx_str and not col_str:
                df = df.T
            elif idx_str and col_str and df.shape[0] > df.shape[1]:
                df = df.T

            try:
                numeric_block = df.values.astype("float32")
            except (ValueError, TypeError):
                df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
                numeric_block = df.values.astype("float32")

            return ad.AnnData(
                X   = sp.csr_matrix(numeric_block),
                obs = pd.DataFrame(index=df.index.astype(str)),
                var = pd.DataFrame(index=df.columns.astype(str)),
            )
        except Exception:
            return None

    def _extract_tarballs(self, gsm_dir):
        for fname in os.listdir(gsm_dir):
            if not (fname.endswith(".tar.gz") or fname.endswith(".tar")):
                continue
            tar_path = os.path.join(gsm_dir, fname)
            try:
                with tarfile.open(tar_path, "r:*") as tf:
                    members = tf.getmembers()
                    if all(os.path.exists(os.path.join(gsm_dir, m.name)) for m in members if m.isfile()):
                        continue
                    print(f"  Extracting {fname} → {gsm_dir}")
                    tf.extractall(path=gsm_dir)
            except Exception as exc:
                print(f"  Warning: could not extract {fname}: {exc}")

    def _find_mtx_dir_canonical(self, root):
        for dirpath, _, filenames in os.walk(root):
            if os.path.basename(dirpath).startswith("_canonical_"):
                continue
            lower = {f.lower() for f in filenames}
            if ("matrix.mtx.gz" in lower
                    and ("features.tsv.gz" in lower or "genes.tsv.gz" in lower)
                    and "barcodes.tsv.gz" in lower):
                return dirpath
        return None

    def _find_and_stage_prefix_named_mtx(self, root, gsm_id):
        import hashlib, shutil

        def _role(fname):
            fl = fname.lower()
            if (fl.endswith(".mtx.gz") or fl.endswith(".mtx")) and "matrix" in fl:
                return "matrix"
            if (fl.endswith(".tsv.gz") or fl.endswith(".tsv")) and "barcodes" in fl:
                return "barcodes"
            if (fl.endswith(".tsv.gz") or fl.endswith(".tsv")) and ("features" in fl or "genes" in fl):
                return "features"
            return None

        def _prefix(fname, role):
            return fname.lower()[:fname.lower().find(role)]

        _CANON = {"matrix": "matrix.mtx.gz", "features": "features.tsv.gz", "barcodes": "barcodes.tsv.gz"}

        def _copy_as_gz(src, dst):
            if src.endswith(".gz"):
                shutil.copy2(src, dst)
            else:
                with open(src, "rb") as fi, gzip.open(dst, "wb") as fo:
                    shutil.copyfileobj(fi, fo)

        groups = {}
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                r = _role(fname)
                if r is None:
                    continue
                key = os.path.join(dirpath, _prefix(fname, r))
                groups.setdefault(key, {})
                if r not in groups[key]:
                    groups[key][r] = os.path.join(dirpath, fname)

        for key, roles in groups.items():
            if not all(r in roles for r in ("matrix", "barcodes", "features")):
                continue
            tag       = hashlib.md5(key.encode()).hexdigest()[:8]
            canon_dir = os.path.join(root, f"_canonical_{tag}")
            if os.path.isdir(canon_dir):
                if all(v in os.listdir(canon_dir) for v in _CANON.values()):
                    return canon_dir
                shutil.rmtree(canon_dir)
            os.makedirs(canon_dir, exist_ok=True)
            for role, src in roles.items():
                _copy_as_gz(src, os.path.join(canon_dir, _CANON[role]))
            print(f"  Staged prefix-named MTX for {gsm_id} → {os.path.relpath(canon_dir, root)}")
            return canon_dir
        return None

    def _find_and_stage_shared_barcodes_features(self, gsm_dir, gse_dir, gsm_id):
        import hashlib, shutil

        _CANON = {"matrix": "matrix.mtx.gz", "features": "features.tsv.gz", "barcodes": "barcodes.tsv.gz"}

        def _copy_as_gz(src, dst):
            if src.endswith(".gz"):
                shutil.copy2(src, dst)
            else:
                with open(src, "rb") as fi, gzip.open(dst, "wb") as fo:
                    shutil.copyfileobj(fi, fo)

        def _is_role(fname, role):
            fl = fname.lower()
            if role == "matrix":
                return "matrix" in fl and (fl.endswith(".mtx.gz") or fl.endswith(".mtx"))
            if role == "barcodes":
                return "barcodes" in fl and (fl.endswith(".tsv.gz") or fl.endswith(".tsv"))
            if role == "features":
                return ("features" in fl or "genes" in fl) and (fl.endswith(".tsv.gz") or fl.endswith(".tsv"))
            return False

        def _find_role(directory, role):
            if not os.path.isdir(directory):
                return None
            return next(
                (os.path.join(directory, f) for f in os.listdir(directory) if _is_role(f, role)),
                None
            )

        matrix_path = matrix_path_any = None
        gsm_lower   = gsm_id.lower()
        for dirpath, _, filenames in os.walk(gsm_dir):
            for fname in filenames:
                if not _is_role(fname, "matrix"):
                    continue
                full = os.path.join(dirpath, fname)
                if gsm_lower in fname.lower():
                    matrix_path = full; break
                if matrix_path_any is None:
                    matrix_path_any = full
            if matrix_path:
                break
        if matrix_path is None and matrix_path_any and os.path.normpath(gsm_dir) != os.path.normpath(gse_dir):
            matrix_path = matrix_path_any
        if matrix_path is None:
            return None

        candidate_dirs = [os.path.dirname(matrix_path)]
        if gsm_dir != candidate_dirs[0]:
            candidate_dirs.append(gsm_dir)
        candidate_dirs.append(gse_dir)
        try:
            for e in os.listdir(gse_dir):
                ep = os.path.join(gse_dir, e)
                if os.path.isdir(ep) and ep not in candidate_dirs:
                    candidate_dirs.append(ep)
        except OSError:
            pass

        barcodes_path = features_path = None
        for cdir in candidate_dirs:
            if not barcodes_path:
                barcodes_path = _find_role(cdir, "barcodes")
            if not features_path:
                features_path = _find_role(cdir, "features")
            if barcodes_path and features_path:
                break

        if barcodes_path is None:
            return None

        tag       = hashlib.md5(matrix_path.encode()).hexdigest()[:8]
        canon_dir = os.path.join(gsm_dir, f"_canonical_shared_{tag}")
        if os.path.isdir(canon_dir):
            if all(v in os.listdir(canon_dir) for v in _CANON.values()):
                return canon_dir

        os.makedirs(canon_dir, exist_ok=True)
        _copy_as_gz(matrix_path,   os.path.join(canon_dir, _CANON["matrix"]))
        _copy_as_gz(barcodes_path, os.path.join(canon_dir, _CANON["barcodes"]))

        if features_path:
            _copy_as_gz(features_path, os.path.join(canon_dir, _CANON["features"]))
            features_note = os.path.relpath(features_path, gse_dir)
        else:
            import scipy.io as _sio
            n_genes = None
            try:
                with gzip.open(matrix_path, "rt") as mf:
                    for line in mf:
                        if line.startswith("%"):
                            continue
                        n_genes = int(line.split()[0]); break
            except Exception:
                pass
            feat_dst = os.path.join(canon_dir, _CANON["features"])
            with gzip.open(feat_dst, "wt") as ff:
                if n_genes:
                    for i in range(1, n_genes + 1):
                        ff.write(f"Gene{i}\tGene{i}\tGene Expression\n")
            features_note = "(synthetic)"

        print(f"  Staged MTX triplet for {gsm_id} → {os.path.relpath(canon_dir, gse_dir)}"
              f"\n    matrix   : {os.path.relpath(matrix_path, gse_dir)}"
              f"\n    barcodes : {os.path.relpath(barcodes_path, gse_dir)}"
              f"\n    features : {features_note}")
        return canon_dir

    def _read_10x_manual(self, mtx_dir):
        import scipy.io as sio2
        files = set(os.listdir(mtx_dir))
        mtx_f = next((f for f in files if f == "matrix.mtx.gz"),   None)
        bar_f = next((f for f in files if f == "barcodes.tsv.gz"),  None)
        gen_f = next((f for f in files if f in ("features.tsv.gz", "genes.tsv.gz")), None)
        if not all([mtx_f, bar_f, gen_f]):
            return None
        try:
            with gzip.open(os.path.join(mtx_dir, mtx_f)) as f:
                X = sio2.mmread(f).T.tocsr()
            with gzip.open(os.path.join(mtx_dir, bar_f), "rt") as f:
                barcodes = [l.strip() for l in f if l.strip()]
            with gzip.open(os.path.join(mtx_dir, gen_f), "rt") as f:
                lines = [l.strip().split("\t") for l in f if l.strip()]
            gene_ids   = [l[0] for l in lines]
            gene_names = [l[1] if len(l) > 1 else l[0] for l in lines]

            if len(set(gene_ids)) == len(gene_ids):
                var_index = gene_ids
            else:
                seen, var_index = {}, []
                for gid in gene_ids:
                    if gid in seen:
                        seen[gid] += 1
                        var_index.append(f"{gid}.{seen[gid]}")
                    else:
                        seen[gid] = 0
                        var_index.append(gid)

            var = pd.DataFrame({"gene_ids": gene_ids, "gene_symbols": gene_names}, index=var_index)
            return ad.AnnData(X=X, obs=pd.DataFrame(index=barcodes), var=var)
        except Exception as exc:
            print(f"    Manual MTX read error: {exc}")
            return None

    def _read_h5_gz(self, file_path):
        import tempfile, shutil
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                tmp_path = tmp.name
            with gzip.open(file_path, "rb") as fi, open(tmp_path, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            adata = None
            for reader in (sc.read_10x_h5, sc.read_hdf5):
                try:
                    adata = reader(tmp_path); break
                except Exception:
                    pass
            if adata is None:
                print(f"    sc readers failed — trying h5py fallback")
                adata = _read_10x_h5_via_h5py(tmp_path)
            if adata is not None:
                adata = _dedup_var_names(adata)
            return adata
        except Exception as exc:
            print(f"    H5.gz decompress failed: {exc}")
            return None
        finally:
            if tmp_path:
                try: os.remove(tmp_path)
                except Exception: pass

    # ── NEW: Tier 0 — GSE-level combined matrix ───────────────────────────

    def _find_gse_level_matrix(self, gse_id, gse_dir, tumor_samples):
        def _is_gse_file(fname, keywords):
            fl = fname.lower()
            return (
                fl.startswith(gse_id.lower())
                and any(k in fl for k in keywords)
                and any(fl.endswith(e) for e in (".tsv.gz", ".csv.gz", ".tsv", ".csv"))
            )

        try:
            all_files = os.listdir(gse_dir)
        except OSError:
            return None

        matrix_kw = ["count", "umi", "expression", "matrix", "expr"]
        meta_kw   = ["cellinfo", "cell_info", "metadata", "meta_data",
                     "barcode", "obs", "sample", "anno", "annotation"]

        search_dir = gse_dir
        nested = os.path.join(gse_dir, gse_id)
        if (not any(_is_gse_file(f, matrix_kw) for f in all_files)
                and os.path.isdir(nested)):
            search_dir = nested
            all_files  = os.listdir(nested)

        matrix_file = next((f for f in all_files if _is_gse_file(f, matrix_kw)), None)
        meta_file   = next((f for f in all_files if _is_gse_file(f, meta_kw)),   None)

        if matrix_file is None:
            return None

        matrix_path = os.path.join(search_dir, matrix_file)
        meta_path   = os.path.join(search_dir, meta_file) if meta_file else None

        print(f"\n  [Tier 0] GSE-level matrix detected : {matrix_file}")
        if meta_path:
            print(f"  [Tier 0] Cell metadata file        : {meta_file}")

        cell_meta = None
        gsm_col   = None
        tumor_set = set(tumor_samples)

        if meta_path:
            try:
                opener = gzip.open(meta_path, "rt") if meta_path.endswith(".gz") \
                         else open(meta_path, "r")
                with opener as f:
                    first = f.readline()
                sep = "\t" if "\t" in first else ","

                if meta_path.endswith(".gz"):
                    with gzip.open(meta_path, "rt") as f:
                        cell_meta = pd.read_csv(f, sep=sep, index_col=0)
                else:
                    cell_meta = pd.read_csv(meta_path, sep=sep, index_col=0)

                for col in cell_meta.columns:
                    vals = cell_meta[col].astype(str)
                    if vals.str.startswith("GSM").any() and vals.isin(tumor_set).any():
                        gsm_col = col
                        break

                if gsm_col is None:
                    idx_vals = cell_meta.index.astype(str)
                    if idx_vals.str.startswith("GSM").any():
                        cell_meta["_gsm_col"] = idx_vals.str.extract(r"(GSM\d+)", expand=False)
                        if cell_meta["_gsm_col"].isin(tumor_set).any():
                            gsm_col = "_gsm_col"

                if gsm_col is None:
                    for col in cell_meta.columns:
                        vals      = cell_meta[col].astype(str)
                        uniq      = set(vals.unique())
                        name_hint = any(k in col.lower() for k in
                                        ("sample","patient","gsm","donor","subject","id","orig"))
                        overlap   = uniq & tumor_set
                        if overlap and (name_hint or len(overlap) / len(uniq) >= 0.5):
                            gsm_col = col
                            break

                if gsm_col is None:
                    sorted_tumor = sorted(tumor_set)
                    for col in cell_meta.columns:
                        vals = cell_meta[col].astype(str)
                        uniq = sorted(vals.unique())
                        if len(uniq) == len(sorted_tumor):
                            mapping = dict(zip(uniq, sorted_tumor))
                            cell_meta["_gsm_col"] = vals.map(mapping)
                            gsm_col = "_gsm_col"
                            print(f"  [Tier 0] Built GSM mapping from column '{col}': "
                                  f"{list(mapping.items())[:3]} …")
                            break

                if gsm_col:
                    print(f"  [Tier 0] GSM ID column found       : '{gsm_col}'")
                else:
                    print("  [Tier 0] WARNING: GSM ID column not found — will use all cells.")

            except Exception as exc:
                print(f"  [Tier 0] Metadata read failed: {exc}")
                cell_meta = None

        print("  [Tier 0] Reading expression matrix (this may take a moment)…")
        adata = self._read_generic_matrix(matrix_path)

        if adata is None:
            print("  [Tier 0] Failed to read GSE-level matrix.")
            return None

        print(f"  [Tier 0] Matrix shape: {adata.shape} (cells × genes)")

        if cell_meta is not None and gsm_col is not None:
            common_idx = cell_meta.index.intersection(adata.obs_names)
            if len(common_idx) == 0:
                print("  [Tier 0] WARNING: metadata index does not match matrix barcodes "
                      "— using all cells.")
            else:
                aligned      = cell_meta.loc[common_idx]
                tumor_mask   = aligned[gsm_col].isin(tumor_set)
                tumor_bc     = aligned.index[tumor_mask]
                print(f"  [Tier 0] Tumor cells identified    : {len(tumor_bc)} / {adata.n_obs}")

                if len(tumor_bc) == 0:
                    print("  [Tier 0] WARNING: 0 tumor cells found — returning full matrix.")
                else:
                    adata = adata[tumor_bc].copy()
                    adata.obs["gsm_id"] = aligned.loc[tumor_bc, gsm_col].astype("category")
                    adata.obs["gse_id"] = gse_id

                    for col in aligned.columns:
                        if col not in ("_gsm_col", gsm_col):
                            try:
                                adata.obs[col] = aligned.loc[tumor_bc, col].values
                            except Exception:
                                pass

                    # CLAUDE EDIT — Tier 0 has no per-GSM GEO metadata text
                    # to scan (that's the per-GSM path's job); instead, look
                    # for a column in the bundled cell-metadata file whose
                    # NAME matches batch_key (case-insensitive
                    # substring), and copy/rename it to batch_key so
                    # Module 2's batch_key can find it consistently.
                    if self.batch_key:
                        if self.batch_key in adata.obs.columns:
                            print(f"  [Tier 0] batch_key '{self.batch_key}' "
                                  f"matched a metadata column of the same name.")
                        else:
                            hint_lower  = self.batch_key.lower()
                            matched_col = next(
                                (c for c in aligned.columns if hint_lower in c.lower()),
                                None
                            )
                            if matched_col:
                                adata.obs[self.batch_key] = aligned.loc[tumor_bc, matched_col].values
                                print(f"  [Tier 0] batch_key '{self.batch_key}' matched "
                                      f"metadata column '{matched_col}' — copied as "
                                      f"'{self.batch_key}'.")
                            else:
                                print(f"  [Tier 0] WARNING: batch_key "
                                      f"'{self.batch_key}' not found in metadata file "
                                      f"columns: {list(aligned.columns)}")
        else:
            print("  [Tier 0] No metadata/GSM mapping — using all cells.")
            adata.obs["gse_id"] = gse_id
            if self.batch_key:
                print(f"  [Tier 0] WARNING: batch_key '{self.batch_key}' requested "
                      f"but no cell-metadata file was found for this GSE-level matrix.")

        return adata

    # ── h5ad builder ──────────────────────────────────────────────────────

    def _build_h5ad(self, gse_id, tumor_samples, save_single=False, batch_hint_map=None):
        if not tumor_samples:
            return None

        gse_dir = os.path.join(self.base_dir, gse_id)
        print("\n========== Reading Tumor Samples ==========")

        adatas = []

        gse_level_adata = self._find_gse_level_matrix(gse_id, gse_dir, tumor_samples)
        if gse_level_adata is not None:
            print("  [Tier 0] Using GSE-level matrix — skipping per-GSM file scan.")
            adatas.append(gse_level_adata)
        else:
            for gsm_id in tumor_samples:

                gsm_dir = os.path.join(gse_dir, gsm_id)

                if not os.path.isdir(gsm_dir):
                    supp_dirs = [
                        d for d in os.listdir(gse_dir)
                        if d.startswith(f"Supp_{gsm_id}")
                           and os.path.isdir(os.path.join(gse_dir, d))
                    ]
                    if supp_dirs:
                        gsm_dir = os.path.join(gse_dir, supp_dirs[0])
                    else:
                        gse_supp_dirs = [
                            os.path.join(gse_dir, d)
                            for d in os.listdir(gse_dir)
                            if os.path.isdir(os.path.join(gse_dir, d))
                               and not d.startswith("Supp_GSM")
                        ]
                        found = None
                        for candidate in [gse_dir] + gse_supp_dirs:
                            try:
                                files_here = os.listdir(candidate)
                            except OSError:
                                continue
                            if any(gsm_id.lower() in f.lower() for f in files_here):
                                found = candidate; break
                        if found:
                            gsm_dir = found
                        else:
                            continue

                self._extract_tarballs(gsm_dir)
                adata = None

                mtx_dir = self._find_mtx_dir_canonical(gsm_dir)
                if mtx_dir:
                    try:
                        print(f"Reading MTX matrix for {gsm_id} (from {os.path.relpath(mtx_dir, gse_dir)})")
                        adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", cache=False)
                    except (SystemExit, Exception) as exc:
                        print(f"  sc.read_10x_mtx failed — trying manual reader for {gsm_id}")
                        adata = self._read_10x_manual(mtx_dir)

                if adata is None:
                    staged = self._find_and_stage_prefix_named_mtx(gsm_dir, gsm_id)
                    if staged:
                        try:
                            print(f"Reading prefix-named MTX for {gsm_id}")
                            adata = sc.read_10x_mtx(staged, var_names="gene_symbols", cache=False)
                        except (SystemExit, Exception):
                            adata = self._read_10x_manual(staged)

                if adata is None:
                    staged = self._find_and_stage_shared_barcodes_features(gsm_dir, gse_dir, gsm_id)
                    if staged:
                        try:
                            print(f"Reading shared-barcodes MTX for {gsm_id}")
                            adata = sc.read_10x_mtx(staged, var_names="gene_symbols", cache=False)
                        except (SystemExit, Exception):
                            adata = self._read_10x_manual(staged)

                if adata is None:
                    for f in os.listdir(gsm_dir):
                        fl = f.lower()
                        if fl.endswith(".tar.gz") or fl.endswith(".tar"):
                            continue
                        if fl.endswith(".mtx") or fl.endswith(".mtx.gz"):
                            continue
                        if (any(fl.endswith(e) for e in (".tsv", ".csv", ".txt", ".gz"))
                                and any(k in fl for k in ("matrix", "counts", "count"))):
                            print(f"Reading generic matrix for {gsm_id}: {f}")
                            adata = self._read_generic_matrix(os.path.join(gsm_dir, f))
                            if adata:
                                break

                if adata is None:
                    for f in sorted(os.listdir(gsm_dir)):
                        fl = f.lower()
                        if fl.endswith(".h5") or fl.endswith(".hdf5"):
                            fp = os.path.join(gsm_dir, f)
                            print(f"Reading H5 file for {gsm_id}: {f}")
                            for reader in (sc.read_10x_h5, sc.read_hdf5):
                                try: adata = reader(fp); break
                                except Exception: pass
                            if adata is None:
                                print(f"  sc readers failed — trying h5py fallback for {gsm_id}")
                                adata = _read_10x_h5_via_h5py(fp)
                            if adata:
                                adata = _dedup_var_names(adata); break

                if adata is None:
                    for f in sorted(os.listdir(gsm_dir)):
                        if f.lower().endswith(".h5.gz"):
                            print(f"Reading H5.gz file for {gsm_id}: {f}")
                            adata = self._read_h5_gz(os.path.join(gsm_dir, f))
                            if adata:
                                break

                if adata is None:
                    for f in os.listdir(gsm_dir):
                        if f.lower().endswith(".loom"):
                            print(f"Reading Loom file for {gsm_id}: {f}")
                            try:
                                adata = sc.read_loom(os.path.join(gsm_dir, f))
                            except Exception as exc:
                                print(f"  Loom read failed: {exc}")
                            if adata:
                                break

                if adata is None:
                    for f in os.listdir(gsm_dir):
                        if f.lower().endswith(".h5ad"):
                            print(f"Reading H5AD file for {gsm_id}: {f}")
                            try:
                                adata = sc.read_h5ad(os.path.join(gsm_dir, f))
                            except Exception as exc:
                                print(f"  H5AD read failed: {exc}")
                            if adata:
                                break

                if adata is None:
                    print(f"Skipping {gsm_id} (no valid expression matrix found)")
                    continue

                adata.obs["gsm_id"] = gsm_id
                adata.obs["gse_id"] = gse_id
                # CLAUDE EDIT — stamp the requested batch_key value
                # (extracted per-GSM in _process_gse) onto every cell from
                # this sample, same pattern as gsm_id/gse_id above.
                if self.batch_key:
                    hint_val = (batch_hint_map or {}).get(gsm_id)
                    adata.obs[self.batch_key] = hint_val if hint_val is not None else "unknown"
                adata.layers["counts"] = adata.X.copy()
                adata.raw = adata
                adata.obs_names_make_unique()
                adata = _dedup_var_names(adata)
                adatas.append(adata)

        if not adatas:
            return None

        try:
            combined = _safe_concat(adatas)
        except Exception as exc:
            print(f"  Warning: concat failed: {exc}\n  Skipping this GSE.")
            return None

        combined.obs_names_make_unique()
        print("... storing 'gsm_id' as categorical")
        print("... storing 'gse_id' as categorical")

        if "gsm_id" in combined.obs.columns:
            combined.obs["gsm_id"] = combined.obs["gsm_id"].astype("category")
        if "gse_id" in combined.obs.columns:
            combined.obs["gse_id"] = combined.obs["gse_id"].astype("category")
        if self.batch_key and self.batch_key in combined.obs.columns:
            combined.obs[self.batch_key] = combined.obs[self.batch_key].astype("category")
            print(f"... storing '{self.batch_key}' (batch_key) as categorical")

        combined.uns["cancer_type"] = self._user_cancer_type
        print(f"... stored cancer_type in h5ad: {self._user_cancer_type}")

        self._store_qc_params(combined)
        combined = self._apply_qc(combined)
        if self.min_genes is not None or self.max_mt is not None:
            print(f"... applied and stored qc_params: min_genes={self.min_genes}, max_mt={self.max_mt}")
        else:
            print("... qc_params not stored (QC disabled — Module 3 will skip QC filtering)")

        if save_single:
            filename = f"{gse_id}_tumor.h5ad"
            combined.write(filename)
            print("\n========== h5ad created ==========")
            print(f"{filename} is created successfully")

        return combined
