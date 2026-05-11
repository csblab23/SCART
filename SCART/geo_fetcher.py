import GEOparse
import os
import tarfile
import scanpy as sc
import anndata as ad
import pandas as pd
import gzip
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ✅ Tabula reference info
TABULA_DOI_LINK = "https://doi.org/10.6084/m9.figshare.27921984"

TABULA_FILES = {
    "bladder_cancer": "Bladder_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "blood_cancer": "Blood_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "bone_marrow_cancer": "Bone_Marrow_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "ear_cancer": "Ear_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "eye_cancer": "Eye_TSP1_30_version2d_10X_smartseq_scvi_Nov122024_updated.h5ad",
    "fat_cancer": "Fat_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "heart_cancer": "Heart_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "kidney_cancer": "Kidney_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "large_intestine_cancer": "Large_Intestine_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "liver_cancer": "Liver_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "lung_cancer": "Lung_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "lymph_node_cancer": "Lymph_Node_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "breast_cancer": "Mammary_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "muscle_cancer": "Muscle_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "ovary_cancer": "Ovary_TSP1_30_version2d_10X_smartseq_scvi_Nov262024.h5ad",
    "pancreas_cancer": "Pancreas_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "prostate_cancer": "Prostate_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "salivary_gland_cancer": "Salivary_Gland_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "skin_cancer": "Skin_TSP1_30_version2d_10X_smartseq_scvi_Nov122024_updated.h5ad",
    "small_intestine_cancer": "Small_Intestine_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "spleen_cancer": "Spleen_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "stomach_cancer": "Stomach_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "testis_cancer": "Testis_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "thymus_cancer": "Thymus_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "tongue_cancer": "Tongue_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "trachea_cancer": "Trachea_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "uterus_cancer": "Uterus_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
    "vasculature_cancer": "Vasculature_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad",
}

# Valid cancer type keys for user reference
VALID_CANCER_TYPES = sorted(TABULA_FILES.keys())

# ── High-priority sample-level metadata fields ────────────────────────────────
_PRIORITY_FIELDS = [
    "characteristics_ch1",
    "source_name_ch1",
    "title",
    "description",
]

# ── GSE series-level fields used to detect study disease context ──────────────
_SERIES_FIELDS = [
    "title",
    "summary",
    "overall_design",
]

# ── Disease / tumor keywords ──────────────────────────────────────────────────
DISEASE_TUMOR_KEYWORDS = [
    # General malignancy
    "tumor", "tumour", "cancer", "carcinoma", "adenocarcinoma",
    "malignant", "malignancy", "metastatic", "metastasis",
    "neoplasm", "neoplastic",

    # Gynaecological / epithelial
    "hgsoc", "lgsoc", "serous ovarian", "ovarian carcinoma",
    "endometrioid carcinoma", "clear cell carcinoma",
    "mucinous carcinoma", "fallopian tube carcinoma",
    "stic", "peritoneal carcinoma", "ascites", "debulking",

    # Haematological
    "leukemia", "leukaemia", "lymphoma", "myeloma",
    "aml", "cml", "all", "cll", "mds",
    "acute myeloid", "chronic myeloid",
    "acute lymphoblastic", "chronic lymphocytic", "acute lymphocytic",
    "t-cell leukemia", "b-cell leukemia",
    "hairy cell leukemia", "large granular lymphocyte",
    "myelodysplastic", "myeloproliferative",
    "polycythemia vera", "essential thrombocythemia", "myelofibrosis",

    # Lymphoid
    "dlbcl", "follicular lymphoma", "mantle cell lymphoma",
    "burkitt lymphoma", "hodgkin", "non-hodgkin",
    "marginal zone lymphoma", "anaplastic large cell",
    "primary mediastinal b-cell",
    "b-cell lymphoma", "t-cell lymphoma",
    "large b-cell", "diffuse large",

    # Plasma cell
    "multiple myeloma", "plasma cell dyscrasia",
    "plasmacytoma", "waldenström",
    "smoldering myeloma", "amyloidosis",
    "myeloma patient", "mm patient",

    # Solid-tumour aliases
    "pdac", "nsclc", "sclc",
    "gbm", "glioblastoma", "glioma", "astrocytoma",
    "melanoma", "sarcoma", "blastoma",
    "hepatocellular", "cholangiocarcinoma",
    "seminoma", "teratoma",
]

# ── Normal keywords for Pass 1 (priority fields only) ────────────────────────
_NORMAL_KEYWORDS = [
    "normal",
    "healthy",
    "healthy control",
    "healthy donor",
    "healthy volunteer",
    "control",
    "adjacent normal",
    "non-tumor",
    "non-tumour",
    "non-cancer",
    "benign",
    "non-malignant",
]

# ── Strict normal keywords for Pass 3 (full metadata blob) ───────────────────
_NORMAL_KEYWORDS_STRICT = [
    "adjacent normal",
    "non-tumor",
    "non-tumour",
    "non-cancer",
    "non-malignant",
    "healthy donor",
    "healthy volunteer",
    "healthy control",
    "normal donor",
    "normal volunteer",
]

# ── Patient-origin indicators for Pass 2 (disease-context rescue) ─────────────
_PATIENT_ORIGIN_TERMS = [
    "patient",
    "pbmc",
    "peripheral blood",
    "bone marrow",
    "blood sample",
    "preinfusion",
    "pre-infusion",
    "post-infusion",
    "biopsy",
    "aspirate",
]


class SampleAnnotator:
    """
    Downloads GEO datasets (or accepts pre-built h5ad files), classifies
    samples as tumour / normal / unspecified, and writes a query h5ad for
    downstream PopV annotation (Module 2).

    Parameters
    ----------
    *inputs : str
        One or more GEO accession IDs (e.g. "GSE158937") or paths to
        existing .h5ad files.

    cancer_type : str
        **Required.**  The cancer type(s) you are studying.

        Two formats are accepted:

        1. **Tabula Sapiens key** – one of the keys in ``TABULA_FILES``
           (e.g. ``"blood_cancer"``, ``"lung_cancer"``).  A matching
           Tabula Sapiens reference file will be recommended for PopV.

        2. **Free-text label** – any string that is *not* a Tabula Sapiens
           key (e.g. ``"brain_cancer"``, ``"thyroid_cancer"``).  The label
           is stored as-is and you will be instructed to supply your own
           reference file for PopV / SCEVAN.

        Multiple types can be provided as a comma-separated string::

            cancer_type="blood_cancer"
            cancer_type="blood_cancer, bone_marrow_cancer"
            cancer_type="my_custom_cancer"

        To see all Tabula Sapiens keys::

            from SCART.geo_fetcher import VALID_CANCER_TYPES
            print(VALID_CANCER_TYPES)

    exclude_gsm_ids : list of str or None, optional
        A list of GSM IDs to exclude from the tumor h5ad, regardless of
        how they were classified.  Excluded IDs are reported in the sample
        summary as "Manually excluded".

        Example::

            annotator = SampleAnnotator(
                "GSE224550",
                cancer_type="blood_cancer",
                exclude_gsm_ids=["GSM7025839", "GSM7025840"],
            )

    min_genes : int or None, optional
        Minimum number of genes detected per cell for QC filtering in
        Module 3. Stored in ``adata.uns['qc_params']``.

    max_mt : float or None, optional
        Maximum mitochondrial gene percentage per cell for QC filtering.
        Stored in ``adata.uns['qc_params']``.

    manual_annotation_col : str or None, optional
        ONLY relevant when providing your own .h5ad file (not a GEO ID).
        Name of the obs column holding cell-type annotations. When provided,
        Module 2 (PopV) is skipped and annotations are used directly in
        Module 3.

    Notes
    -----
    Sample classification logic (three-pass, context-aware)
    -------------------------------------------------------
    Pass 1 — HIGH-PRIORITY FIELDS
        Checks characteristics_ch1, source_name_ch1, title, description.
        a. Normal keyword  → normal
        b. Tumor keyword   → tumor

    Pass 2 — GSE DISEASE CONTEXT RESCUE (only when Pass 1 inconclusive)
        If the series IS a cancer study AND the sample's priority text
        contains a patient-origin term → tumor.

    Pass 3 — FULL METADATA BLOB
        a. Strict normal keyword → normal
        b. Tumor keyword         → tumor

    Pass 4 — unspecified

    MTX reading strategy (three-tier)
    ----------------------------------
    GEO datasets package their 10x MTX triplets in several different ways.
    The reader attempts three strategies in order:

    Tier 1 — Canonical layout
        Walk the GSM directory tree looking for a folder that already
        contains all three canonical files (matrix.mtx / matrix.mtx.gz,
        features/genes .tsv, barcodes .tsv).  If found, read directly
        with sc.read_10x_mtx.

    Tier 2 — Prefix-named layout (e.g. GSE176078, GSE161529)
        Many datasets ship MTX files with a sample-ID or cohort prefix:
            GSM5354513_CID3586_matrix.mtx.gz
            GSM5354513_CID3586_barcodes.tsv.gz
            GSM5354513_CID3586_features.tsv.gz
        or flat in the GSM directory:
            GSM4909278_B1-MH0033-matrix.mtx.gz
            GSM4909278_B1-MH0033-barcodes.tsv.gz
            GSM4909278_B1-MH0033-features.tsv.gz
        _find_mtx_dir misses these because it checks for "barcodes" in the
        filename, but the prefix obscures the pattern at the directory level
        — actually _find_mtx_dir does match these correctly.  The real
        problem is that sc.read_10x_mtx requires EXACTLY the canonical
        names.  _rename_mtx_files renames them, but only if the target
        does NOT already exist.  On re-runs where a previous partial rename
        left stale files, or when the tarball puts all three files in the
        same directory as a flat dump, this silently fails.

        Tier 2 fixes this by:
          a. Walking the entire GSM tree to collect every file whose name
             contains "matrix", "features"/"genes", and "barcodes" (with
             any prefix and any separator).
          b. Grouping them by their shared prefix (everything before the
             role keyword).
          c. For each matched group, copying (not renaming — safe on
             re-runs) the three files to a clean temp subdirectory with
             canonical names and reading from there.

    Tier 3 — Generic CSV/TSV matrix
        Last resort for non-standard flat matrix files.
    """

    def __init__(
        self,
        *inputs,
        cancer_type: str,
        exclude_gsm_ids: list = None,
        min_genes: int = None,
        max_mt: float = None,
        manual_annotation_col: str = None,
    ):
        self.inputs    = list(inputs)
        self.base_dir  = "GSE_data"
        self.min_genes = min_genes
        self.max_mt    = max_mt
        self.exclude_gsm_ids = set(exclude_gsm_ids) if exclude_gsm_ids else set()

        if not cancer_type or not isinstance(cancer_type, str):
            raise ValueError(
                "\ncancer_type is required.\n\n"
                "Provide a Tabula Sapiens key, e.g.:\n"
                "  cancer_type='blood_cancer'\n\n"
                "Or a free-text label for cancers not in Tabula Sapiens:\n"
                "  cancer_type='brain_cancer'\n\n"
                "To see all Tabula Sapiens keys:\n"
                "  from SCART.geo_fetcher import VALID_CANCER_TYPES\n"
                "  print(VALID_CANCER_TYPES)"
            )

        self._user_cancer_type, self._tabula_types, self._unknown_types = (
            self._parse_cancer_type(cancer_type)
        )

        self.manual_annotation_col = manual_annotation_col
        os.makedirs(self.base_dir, exist_ok=True)

        self.gse_ids     = []
        self.h5ad_inputs = []

        for item in self.inputs:
            if isinstance(item, str) and item.lower().endswith(".h5ad"):
                self.h5ad_inputs.append(item)
            else:
                self.gse_ids.append(item)

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _parse_cancer_type(self, cancer_type: str):
        tokens        = [t.strip() for t in cancer_type.split(",") if t.strip()]
        tabula_types  = [t for t in tokens if t in TABULA_FILES]
        unknown_types = [t for t in tokens if t not in TABULA_FILES]
        return ", ".join(tokens), tabula_types, unknown_types

    def _store_qc_params(self, adata):
        if self.min_genes is None and self.max_mt is None:
            adata.uns.pop("qc_params", None)
            return
        adata.uns["qc_params"] = {"min_genes": self.min_genes, "max_mt": self.max_mt}

    def _store_manual_annotation(self, adata, source_file: str):
        col = self.manual_annotation_col

        if col not in adata.obs.columns:
            raise ValueError(
                f"\nmanual_annotation_col='{col}' not found in adata.obs "
                f"of '{source_file}'.\n"
                f"Available obs columns: {list(adata.obs.columns)}\n\n"
                "Please check the column name and try again."
            )

        if "popv_majority_vote_prediction" in adata.obs.columns:
            print(f"  WARNING: 'popv_majority_vote_prediction' already exists — overwriting with '{col}'.")

        adata.obs["popv_majority_vote_prediction"] = adata.obs[col].astype(str)
        adata.uns["manual_annotation_col"]         = col
        adata.uns["skip_popv"]                     = True

        unique_labels  = sorted(adata.obs["popv_majority_vote_prediction"].unique())
        epithelial     = [l for l in unique_labels if "epithelial cell" in l.lower()]
        non_epithelial = [l for l in unique_labels if "epithelial cell" not in l.lower()]

        print(f"\n  Manual annotation column : '{col}'")
        print(f"  Copied to               : 'popv_majority_vote_prediction'")
        print(f"  Total unique labels     : {len(unique_labels)}")
        print(f"  Epithelial labels found : {epithelial if epithelial else 'NONE — check your label names!'}")
        print(f"  Non-epithelial labels   : {non_epithelial}")
        print( "  PopV will be SKIPPED for this file (adata.uns['skip_popv'] = True)")

        if not epithelial:
            print(
                "\n  ⚠ WARNING: No epithelial labels detected.\n"
                "  Module 3 looks for labels ending with 'epithelial cell' (case-insensitive).\n"
                f"  Your labels in '{col}' do not match this pattern.\n"
                "  Please rename your epithelial label(s) to end with 'epithelial cell'."
            )

    def _print_reference_guidance(self):
        print("\n========== REFERENCE GUIDANCE ==========")
        print(f"Cancer type(s) provided: {self._user_cancer_type}\n")

        if self.h5ad_inputs:
            if self.manual_annotation_col:
                print("👉 You provided your own h5ad file WITH manual annotations.")
                print("👉 PopV (Module 2) will be SKIPPED automatically.")
                print("👉 Proceed directly to Module 3 (preprocessing).")
            else:
                print("👉 You provided your own h5ad file.")
                print("👉 Please provide your own reference file for PopV.")

        for ct in self._tabula_types:
            print(f"\n✅ Tabula Sapiens reference available: {ct}")
            print(f"   Download : {TABULA_FILES[ct]}")
            print(f"   From     : {TABULA_DOI_LINK}")
            print( "   Use this reference in the next module (PopV).")

        for ct in self._unknown_types:
            print(f"\n⚠️  '{ct}' is not available in Tabula Sapiens.")
            print( "   ❌ No built-in reference file for this cancer type.")
            print( "   👉 Please provide your own reference file for PopV / SCEVAN.")

    # ──────────────────────────────────────────────────────────────────────
    # Text extraction
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_priority_text(gsm) -> str:
        parts = []
        for field in _PRIORITY_FIELDS:
            vals = gsm.metadata.get(field, [])
            if vals:
                parts.append(" ".join(str(v) for v in vals))
        return " ".join(parts).lower()

    @staticmethod
    def _extract_full_text(gsm) -> str:
        return " ".join(str(v) for v in gsm.metadata.values()).lower()

    @staticmethod
    def _extract_series_disease_context(gse) -> str:
        parts = []
        for field in _SERIES_FIELDS:
            vals = gse.metadata.get(field, [])
            if vals:
                parts.append(" ".join(str(v) for v in vals))
        return " ".join(parts).lower()

    # ──────────────────────────────────────────────────────────────────────
    # Classification
    # ──────────────────────────────────────────────────────────────────────

    def _classify_gsm(self, gsm, series_is_cancer: bool) -> str:
        priority_text = self._extract_priority_text(gsm)

        if any(k in priority_text for k in _NORMAL_KEYWORDS):
            return "normal"
        if any(k in priority_text for k in DISEASE_TUMOR_KEYWORDS):
            return "tumor"

        if series_is_cancer and any(k in priority_text for k in _PATIENT_ORIGIN_TERMS):
            return "tumor"

        full_text = self._extract_full_text(gsm)
        if any(k in full_text for k in _NORMAL_KEYWORDS_STRICT):
            return "normal"
        if any(k in full_text for k in DISEASE_TUMOR_KEYWORDS):
            return "tumor"

        return "unspecified"

    # ──────────────────────────────────────────────────────────────────────
    # Public entry-point
    # ──────────────────────────────────────────────────────────────────────

    def run(self):
        normal          = []
        tumor           = []
        unspecified     = []
        annotation_info = {}
        tumor_adatas    = []
        results         = {}

        for gse_id in self.gse_ids:
            n, t, u, ann = self._process_gse(gse_id)
            normal.extend(n)
            tumor.extend(t)
            unspecified.extend(u)
            annotation_info.update(ann)

            adata = self._build_h5ad(
                gse_id, t,
                save_single=(len(self.gse_ids) == 1 and len(self.h5ad_inputs) == 0),
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
            tumor_adatas.append(adata)
            results[file] = ([], [], [], {}, None, self._user_cancer_type)

        query_h5ad   = None
        total_inputs = len(self.gse_ids) + len(self.h5ad_inputs)

        if total_inputs == 1:
            if len(self.gse_ids) == 1:
                query_h5ad = f"{self.gse_ids[0]}_tumor.h5ad"
                results[self.gse_ids[0]] = (
                    normal, tumor, unspecified,
                    annotation_info, query_h5ad, self._user_cancer_type
                )

            elif len(self.h5ad_inputs) == 1:
                adata    = tumor_adatas[0]
                filename = "input_tumor.h5ad"
                adata.write(filename)
                print("\n========== h5ad created ==========")
                print(f"{filename} is created successfully")

                if self.manual_annotation_col is not None:
                    print(f"Manual annotation stored → col='{self.manual_annotation_col}', skip_popv=True")
                    print("Next step: run Module 3 (preprocessing) directly.")

                if self.min_genes is not None or self.max_mt is not None:
                    print(f"QC params stored → min_genes={self.min_genes}, max_mt={self.max_mt}")
                else:
                    print("QC step disabled (no min_genes / max_mt provided — will be skipped in Module 3)")

                query_h5ad   = filename
                key          = self.h5ad_inputs[0]
                results[key] = ([], [], [], {}, query_h5ad, self._user_cancer_type)

        elif total_inputs > 1 and len(tumor_adatas) > 0:
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
                    print(f"\n  Combined h5ad: manual annotation carried from '{self.manual_annotation_col}'.")
                else:
                    print("\n  WARNING: 'popv_majority_vote_prediction' lost during concat.")

            self._store_qc_params(combined)
            combined.write("combined_tumor.h5ad")
            print("\n========== h5ad created ==========")
            print("combined_tumor.h5ad is created successfully")

            if self.manual_annotation_col is not None:
                print(f"Manual annotation stored → col='{self.manual_annotation_col}', skip_popv=True")
                print("Next step: run Module 3 (preprocessing) directly.")

            if self.min_genes is not None or self.max_mt is not None:
                print(f"QC params stored → min_genes={self.min_genes}, max_mt={self.max_mt}")
            else:
                print("QC step disabled (no min_genes / max_mt provided — will be skipped in Module 3)")

            query_h5ad = "combined_tumor.h5ad"
            for key in results:
                n, t, u, ann, _, ct = results[key]
                results[key]        = (n, t, u, ann, query_h5ad, ct)

        self._print_reference_guidance()

        return (
            normal, tumor, unspecified, annotation_info,
            query_h5ad, self._user_cancer_type, results
        )

    # ──────────────────────────────────────────────────────────────────────
    # GEO processing
    # ──────────────────────────────────────────────────────────────────────

    def _process_gse(self, gse_id):
        gse_dir = os.path.join(self.base_dir, gse_id)
        os.makedirs(gse_dir, exist_ok=True)

        gse = GEOparse.get_GEO(geo=gse_id, destdir=gse_dir)
        gse.download_supplementary_files(gse_dir)

        series_text      = self._extract_series_disease_context(gse)
        series_is_cancer = any(k in series_text for k in DISEASE_TUMOR_KEYWORDS)

        normal             = []
        tumor              = []
        unspecified        = []
        manually_excluded  = []
        annotation_info    = {}
        excluded_non_scrna = []
        excluded_non_human = []

        for gsm_id, gsm in gse.gsms.items():

            organism = " ".join(gsm.metadata.get("organism_ch1", [])).lower()
            if "homo sapiens" not in organism:
                excluded_non_human.append(gsm_id)
                continue

            library = " ".join(gsm.metadata.get("library_strategy", [])).lower()
            if not any(k in library for k in ["rna-seq", "scrna", "single cell"]):
                excluded_non_scrna.append(gsm_id)
                continue

            if gsm_id in self.exclude_gsm_ids:
                manually_excluded.append(gsm_id)
                annotation_info[gsm_id] = "manually_excluded"
                continue

            label = self._classify_gsm(gsm, series_is_cancer)
            annotation_info[gsm_id] = label

            if label == "normal":
                normal.append(gsm_id)
            elif label == "tumor":
                tumor.append(gsm_id)
            else:
                unspecified.append(gsm_id)

        print(f"\n========== SAMPLE SUMMARY: {gse_id} ==========")
        print(f"Cancer type (user-supplied): {self._user_cancer_type}")
        print("Normal samples:",       ", ".join(normal)            if normal            else "None")
        print("Tumor samples:",        ", ".join(tumor)             if tumor             else "None")
        print("Unspecified samples:",  ", ".join(unspecified)       if unspecified       else "None")
        print("Manually excluded:",    ", ".join(manually_excluded) if manually_excluded else "None")
        print("Excluded (non-human):", ", ".join(excluded_non_human)  if excluded_non_human  else "None")
        print("Excluded (non-scRNA):", ", ".join(excluded_non_scrna)  if excluded_non_scrna  else "None")

        return normal, tumor, unspecified, annotation_info

    # ──────────────────────────────────────────────────────────────────────
    # File I/O helpers
    # ──────────────────────────────────────────────────────────────────────

    def _read_generic_matrix(self, file_path):
        try:
            if file_path.endswith(".gz"):
                with gzip.open(file_path, 'rt') as f:
                    df = pd.read_csv(f, sep=None, engine='python')
            else:
                df = pd.read_csv(file_path, sep=None, engine='python')

            if df.shape[0] < df.shape[1]:
                df = df.T

            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            return ad.AnnData(df)
        except Exception:
            return None

    def _extract_tarballs(self, gsm_dir: str):
        for fname in os.listdir(gsm_dir):
            if not (fname.endswith(".tar.gz") or fname.endswith(".tar")):
                continue
            tar_path = os.path.join(gsm_dir, fname)
            try:
                with tarfile.open(tar_path, "r:*") as tf:
                    members     = tf.getmembers()
                    all_present = all(
                        os.path.exists(os.path.join(gsm_dir, m.name))
                        for m in members if m.isfile()
                    )
                    if all_present:
                        continue
                    print(f"  Extracting {fname} → {gsm_dir}")
                    tf.extractall(path=gsm_dir)
            except Exception as exc:
                print(f"  Warning: could not extract {fname}: {exc}")

    def _find_mtx_dir(self, root: str):
        """
        Walk *root* recursively and return the first directory that already
        contains all three 10x MTX files with CANONICAL names
        (matrix.mtx / matrix.mtx.gz, features/genes .tsv, barcodes .tsv).

        This is Tier 1 — fast path for datasets that already ship canonical
        names or have been successfully renamed on a previous run.
        """
        for dirpath, dirnames, filenames in os.walk(root):
            lower        = [f.lower() for f in filenames]
            has_matrix   = any(f in ("matrix.mtx.gz", "matrix.mtx") for f in lower)
            has_features = any(f in ("features.tsv.gz", "features.tsv",
                                     "genes.tsv.gz",    "genes.tsv")   for f in lower)
            has_barcodes = any(f in ("barcodes.tsv.gz", "barcodes.tsv") for f in lower)
            if has_matrix and has_features and has_barcodes:
                return dirpath
        return None

    def _rename_mtx_files(self, mtx_dir: str):
        """
        Rename files in *mtx_dir* to canonical names expected by
        sc.read_10x_mtx.  Skips if the canonical target already exists
        to avoid clobbering on re-runs.
        """
        for fname in os.listdir(mtx_dir):
            src = os.path.join(mtx_dir, fname)
            fl  = fname.lower()
            if "matrix" in fl and (fl.endswith(".mtx.gz") or fl.endswith(".mtx")):
                dst_name = "matrix.mtx.gz" if fl.endswith(".gz") else "matrix.mtx"
            elif ("features" in fl or "genes" in fl) and (fl.endswith(".tsv.gz") or fl.endswith(".tsv")):
                dst_name = "features.tsv.gz" if fl.endswith(".gz") else "features.tsv"
            elif "barcodes" in fl and (fl.endswith(".tsv.gz") or fl.endswith(".tsv")):
                dst_name = "barcodes.tsv.gz" if fl.endswith(".gz") else "barcodes.tsv"
            else:
                continue
            dst = os.path.join(mtx_dir, dst_name)
            if src != dst and not os.path.exists(dst):
                try:
                    os.rename(src, dst)
                except Exception:
                    pass

    def _find_and_stage_prefix_named_mtx(self, root: str, gsm_id: str):
        """
        Tier 2 MTX reader — handles prefix-named MTX triplets.

        GEO datasets frequently ship 10x files with a sample-ID or cohort
        prefix instead of the canonical bare names, e.g.:

            GSM5354513_CID3586_matrix.mtx.gz       ← inside CID3586/ subdir
            GSM5354513_CID3586_barcodes.tsv.gz
            GSM5354513_CID3586_features.tsv.gz

        or flat in the GSM supplementary directory:

            GSM4909278_B1-MH0033-matrix.mtx.gz
            GSM4909278_B1-MH0033-barcodes.tsv.gz
            GSM4909278_B1-MH0033-features.tsv.gz

        sc.read_10x_mtx requires EXACTLY the canonical bare names, so these
        cannot be read directly.  This method:

          1. Walks the entire GSM directory tree collecting every file whose
             lowercase name contains the role keywords "matrix", "barcodes",
             and "features" / "genes" with valid extensions.

          2. Groups the collected files by their shared prefix string
             (everything in the filename before the role keyword), ensuring
             that only files that truly belong to the same triplet are
             grouped together.

          3. For each complete triplet group, copies the three files into a
             fresh canonical subdirectory (``_canonical_<hash>/``) using the
             exact names sc.read_10x_mtx expects, then returns that directory
             path.

        Using copy (shutil.copy2) rather than rename is safe on re-runs —
        if the canonical directory already exists and is complete, it is
        returned immediately without re-copying.

        Parameters
        ----------
        root : str
            Root directory to search (the GSM supplementary directory).
        gsm_id : str
            The GSM accession string, used for informative print messages.

        Returns
        -------
        str or None
            Path to a directory containing canonical MTX triplet files,
            or None if no complete triplet was found.
        """
        import shutil
        import hashlib

        # ── Role detection helpers ─────────────────────────────────────────
        def _role(fname: str):
            """Return 'matrix' | 'features' | 'barcodes' | None."""
            fl = fname.lower()
            if not (fl.endswith(".mtx.gz") or fl.endswith(".mtx")
                    or fl.endswith(".tsv.gz") or fl.endswith(".tsv")):
                return None
            if fl.endswith(".mtx.gz") or fl.endswith(".mtx"):
                if "matrix" in fl:
                    return "matrix"
            if fl.endswith(".tsv.gz") or fl.endswith(".tsv"):
                if "barcodes" in fl:
                    return "barcodes"
                if "features" in fl or "genes" in fl:
                    return "features"
            return None

        def _prefix(fname: str, role: str) -> str:
            """
            Extract the prefix part of the filename before the role keyword.
            E.g. "GSM5354513_CID3586_matrix.mtx.gz" → "gsm5354513_cid3586_"
            """
            fl = fname.lower()
            idx = fl.find(role)
            return fl[:idx]

        # ── Collect all candidate files ────────────────────────────────────
        # Map: prefix → {role: absolute_path}
        groups: dict = {}

        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                role = _role(fname)
                if role is None:
                    continue
                pre  = _prefix(fname, role)
                key  = os.path.join(dirpath, pre)   # unique per (dir, prefix)
                if key not in groups:
                    groups[key] = {}
                # Keep first match per role (avoid duplicates from re-runs)
                if role not in groups[key]:
                    groups[key][role] = os.path.join(dirpath, fname)

        # ── Find a complete triplet ────────────────────────────────────────
        for key, roles in groups.items():
            if not ("matrix" in roles and "barcodes" in roles and "features" in roles):
                continue

            # Build a stable canonical directory name from the key hash
            tag      = hashlib.md5(key.encode()).hexdigest()[:8]
            canon_dir = os.path.join(root, f"_canonical_{tag}")

            # Canonical target filenames
            def _canon_name(src_path: str, role: str) -> str:
                ext = ".gz" if src_path.endswith(".gz") else ""
                if role == "matrix":
                    return f"matrix.mtx{ext}"
                if role == "features":
                    return f"features.tsv{ext}"
                return f"barcodes.tsv{ext}"

            # Check if already staged and complete
            if os.path.isdir(canon_dir):
                staged = os.listdir(canon_dir)
                if len(staged) >= 3:
                    return canon_dir    # already complete from a prior run

            os.makedirs(canon_dir, exist_ok=True)

            # Copy each file to its canonical name
            for role, src_path in roles.items():
                dst_name = _canon_name(src_path, role)
                dst_path = os.path.join(canon_dir, dst_name)
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)

            print(f"  Staged prefix-named MTX triplet → {os.path.relpath(canon_dir, root)}")
            return canon_dir

        return None

    # ──────────────────────────────────────────────────────────────────────

    def _build_h5ad(self, gse_id, tumor_samples, save_single=False):
        if len(tumor_samples) == 0:
            return None

        gse_dir = os.path.join(self.base_dir, gse_id)
        print("\n========== Reading Tumor Samples ==========")
        adatas = []

        for gsm_id in tumor_samples:
            gsm_dir = os.path.join(gse_dir, gsm_id)

            if not os.path.isdir(gsm_dir):
                supp_dirs = [
                    d for d in os.listdir(gse_dir)
                    if d.startswith(f"Supp_{gsm_id}")
                ]
                if supp_dirs:
                    gsm_dir = os.path.join(gse_dir, supp_dirs[0])
                else:
                    continue

            # Extract any tarballs first
            self._extract_tarballs(gsm_dir)

            adata = None

            # ── Tier 1: canonical layout ───────────────────────────────────
            # Try to find a directory that already has canonical filenames,
            # optionally after renaming prefix-free files.
            mtx_dir = self._find_mtx_dir(gsm_dir)
            if mtx_dir is not None:
                self._rename_mtx_files(mtx_dir)
                # Re-check after rename (in case rename just completed it)
                mtx_dir = self._find_mtx_dir(gsm_dir)

            if mtx_dir is not None:
                try:
                    print(f"Reading MTX matrix for {gsm_id} "
                          f"(from {os.path.relpath(mtx_dir, gse_dir)})")
                    adata = sc.read_10x_mtx(
                        mtx_dir, var_names="gene_symbols", cache=False
                    )
                except Exception as exc:
                    print(f"  MTX read failed for {gsm_id}: {exc}")

            # ── Tier 2: prefix-named layout ────────────────────────────────
            # Handles files like GSM5354513_CID3586_matrix.mtx.gz (GSE176078)
            # or GSM4909278_B1-MH0033-matrix.mtx.gz (GSE161529) by staging
            # them into a canonical temp directory.
            if adata is None:
                staged_dir = self._find_and_stage_prefix_named_mtx(gsm_dir, gsm_id)
                if staged_dir is not None:
                    try:
                        print(f"Reading prefix-named MTX for {gsm_id} "
                              f"(staged at {os.path.relpath(staged_dir, gse_dir)})")
                        adata = sc.read_10x_mtx(
                            staged_dir, var_names="gene_symbols", cache=False
                        )
                    except Exception as exc:
                        print(f"  Staged MTX read failed for {gsm_id}: {exc}")

            # ── Tier 3: generic CSV/TSV matrix ─────────────────────────────
            if adata is None:
                files = os.listdir(gsm_dir)
                for f in files:
                    fl = f.lower()
                    if fl.endswith(".tar.gz") or fl.endswith(".tar"):
                        continue
                    if (any(fl.endswith(ext) for ext in [".tsv", ".csv", ".txt", ".gz"])
                            and "matrix" in fl):
                        file_path = os.path.join(gsm_dir, f)
                        print(f"Reading generic matrix for {gsm_id}: {f}")
                        adata = self._read_generic_matrix(file_path)
                        if adata is not None:
                            break

            if adata is None:
                print(f"Skipping {gsm_id} (no valid expression matrix found)")
                continue

            adata.obs["gsm_id"]    = gsm_id
            adata.obs["gse_id"]    = gse_id
            adata.layers["counts"] = adata.X.copy()
            adata.raw              = adata
            adata.obs_names_make_unique()
            adatas.append(adata)

        if len(adatas) == 0:
            return None

        combined = ad.concat(adatas, join="outer")
        combined.obs_names_make_unique()

        print("... storing 'gsm_id' as categorical")
        print("... storing 'gse_id' as categorical")

        combined.obs["gsm_id"]      = combined.obs["gsm_id"].astype("category")
        combined.obs["gse_id"]      = combined.obs["gse_id"].astype("category")
        combined.uns["cancer_type"] = self._user_cancer_type
        print(f"... stored cancer_type in h5ad: {self._user_cancer_type}")

        self._store_qc_params(combined)

        if self.min_genes is not None or self.max_mt is not None:
            print(f"... stored qc_params in h5ad: min_genes={self.min_genes}, max_mt={self.max_mt}")
        else:
            print("... qc_params not stored (QC disabled — Module 3 will skip QC filtering)")

        if save_single:
            filename = f"{gse_id}_tumor.h5ad"
            combined.write(filename)
            print("\n========== h5ad created ==========")
            print(f"{filename} is created successfully")

        return combined
