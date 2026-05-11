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
# Used in Pass 1: fields that directly describe what the sample IS.
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
# NOTE: "pbmc", "donor", "patient" are intentionally ABSENT.
#   - "pbmc" = cell isolation method, used in cancer patients AND healthy donors
#   - "donor" = ambiguous (could be cancer patient donating cells for therapy)
#   - "patient" = demographic, not a disease-state indicator
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
# Much narrower — avoids matching "normal tissue" in cancer study abstracts
# (e.g. "we compared tumor vs normal ovary in HGSOC patients").
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
# When a study IS a cancer study and a sample contains one of these terms,
# the sample is classified as tumor — because it is a patient-derived sample
# even if the word "tumor" does not appear in its individual GSM metadata.
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
        how they were classified.  Use this when you have inspected the
        sample summary and want to drop specific samples before the h5ad
        is written — for example, to remove CAR-T cell product samples
        from a therapy study, or to drop samples with known quality issues.

        The excluded IDs are reported in the sample summary as
        "Manually excluded" so the decision is fully traceable.

        Example::

            annotator = SampleAnnotator(
                "GSE224550",
                cancer_type="blood_cancer",
                exclude_gsm_ids=[
                    "GSM7025839", "GSM7025840",   # CAR-T products
                    "GSM7025847", "GSM7025848",
                ],
            )

        If not provided (default: None), all classified tumor samples are
        included in the h5ad.

    min_genes : int or None, optional
        Minimum number of genes detected per cell for QC filtering in
        Module 3. Stored in ``adata.uns['qc_params']``.
        If not provided (default: None), the QC step is skipped in Module 3.

    max_mt : float or None, optional
        Maximum mitochondrial gene percentage per cell for QC filtering.
        Stored in ``adata.uns['qc_params']``.
        If not provided (default: None), the QC step is skipped in Module 3.

    manual_annotation_col : str or None, optional
        ONLY relevant when providing your own .h5ad file (not a GEO ID).
        Name of the obs column holding cell-type annotations. When provided,
        Module 2 (PopV) is skipped and annotations are used directly in
        Module 3.

        Requirements for the annotation column
        ---------------------------------------
        - Must exist in adata.obs of every h5ad file you pass.
        - Must contain string cell-type labels for every cell.
        - Epithelial cell labels must end with "epithelial cell"
          (case-insensitive) for Module 3 to detect them correctly.
        - The column is copied into "popv_majority_vote_prediction" so
          the downstream pipeline can find it without any changes.

        If not provided (default: None), PopV runs normally on the h5ad.

    Notes
    -----
    GEO ID inputs always run the full PopV pipeline regardless of
    manual_annotation_col — the parameter is silently ignored for GEO IDs.

    Sample classification logic (four-pass, context-aware)
    -------------------------------------------------------
    Pass 1 — HIGH-PRIORITY FIELDS
        Checks ``characteristics_ch1``, ``source_name_ch1``, ``title``,
        and ``description`` — the fields that directly describe the sample.
        a. Normal keyword match  → **normal**
        b. Tumor keyword match   → **tumor**

    Pass 2 — GSE DISEASE CONTEXT RESCUE
        Applied only when Pass 1 is inconclusive.  The GSE series
        title/summary/overall_design is checked ONCE for disease keywords
        to determine if the overall study is a cancer study.  If it IS a
        cancer study AND the sample's priority text contains a
        patient-origin term (e.g. "patient", "pbmc", "peripheral blood",
        "bone marrow", "preinfusion") → **tumor**.

        This pass correctly handles haematological cancer datasets where
        blood/PBMC samples from cancer patients lack the word "tumor" in
        their individual GSM metadata (e.g. "cell type: pbmc",
        "patient: P1", "time: preinfusion").

    Pass 3 — FULL METADATA BLOB
        a. Strict normal keyword → **normal**
        b. Tumor keyword         → **tumor**

    Pass 4 — **unspecified**

    Why "pbmc" is NOT a normal keyword
    ------------------------------------
    PBMC is a cell isolation method, not a disease-state descriptor.
    PBMCs are collected from healthy donors AND cancer patients alike.
    The GSE disease context (Pass 2) handles PBMC samples correctly by
    asking whether the study is a cancer study.

    Manual exclusion
    ----------------
    No samples are automatically excluded based on cell type.  If your
    dataset contains samples you do not want in the tumor h5ad (e.g.
    CAR-T cell products, infusion products, quality-failed samples),
    pass their IDs via ``exclude_gsm_ids`` and they will be skipped.
    The full classification still runs on every sample so you can inspect
    the summary before deciding what to exclude.
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

        # Normalise exclusion list to a set for O(1) lookup
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
        """Lowercase text from sample-specific priority fields only."""
        parts = []
        for field in _PRIORITY_FIELDS:
            vals = gsm.metadata.get(field, [])
            if vals:
                parts.append(" ".join(str(v) for v in vals))
        return " ".join(parts).lower()

    @staticmethod
    def _extract_full_text(gsm) -> str:
        """Lowercase text from ALL metadata fields of a GSM."""
        return " ".join(str(v) for v in gsm.metadata.values()).lower()

    @staticmethod
    def _extract_series_disease_context(gse) -> str:
        """Lowercase text from GSE series-level fields (title, summary, design)."""
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
        """
        Classify a single GSM as 'normal', 'tumor', or 'unspecified'.

        Three-pass, context-aware algorithm
        ------------------------------------
        Pass 1 — HIGH-PRIORITY FIELDS (characteristics_ch1, source_name_ch1,
                  title, description):
            a. Normal keyword  → normal
            b. Tumor keyword   → tumor

        Pass 2 — GSE DISEASE CONTEXT RESCUE (only when Pass 1 inconclusive):
            If the series IS a cancer study AND the sample's priority text
            contains a patient-origin term → tumor.

        Pass 3 — FULL METADATA BLOB (strict normal / any tumor keyword):
            a. Strict normal keyword → normal
            b. Tumor keyword         → tumor

        Pass 4 — unspecified
        """
        priority_text = self._extract_priority_text(gsm)

        # Pass 1: high-priority sample fields
        if any(k in priority_text for k in _NORMAL_KEYWORDS):
            return "normal"
        if any(k in priority_text for k in DISEASE_TUMOR_KEYWORDS):
            return "tumor"

        # Pass 2: GSE disease context rescue
        if series_is_cancer and any(k in priority_text for k in _PATIENT_ORIGIN_TERMS):
            return "tumor"

        # Pass 3: full metadata blob with strict keywords
        full_text = self._extract_full_text(gsm)
        if any(k in full_text for k in _NORMAL_KEYWORDS_STRICT):
            return "normal"
        if any(k in full_text for k in DISEASE_TUMOR_KEYWORDS):
            return "tumor"

        # Pass 4: inconclusive
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
        """
        Download a GEO series, classify each GSM, and return lists of
        normal / tumor / unspecified sample IDs.

        Samples listed in self.exclude_gsm_ids are skipped after
        classification and reported separately in the summary.
        """
        gse_dir = os.path.join(self.base_dir, gse_id)
        os.makedirs(gse_dir, exist_ok=True)

        gse = GEOparse.get_GEO(geo=gse_id, destdir=gse_dir)
        gse.download_supplementary_files(gse_dir)

        # Build series disease context ONCE for the whole GSE
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

            # ── Filter: human only ─────────────────────────────────────────
            organism = " ".join(gsm.metadata.get("organism_ch1", [])).lower()
            if "homo sapiens" not in organism:
                excluded_non_human.append(gsm_id)
                continue

            # ── Filter: scRNA-seq only ─────────────────────────────────────
            library = " ".join(gsm.metadata.get("library_strategy", [])).lower()
            if not any(k in library for k in ["rna-seq", "scrna", "single cell"]):
                excluded_non_scrna.append(gsm_id)
                continue

            # ── Manual exclusion check ─────────────────────────────────────
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
        print("Normal samples:",      ", ".join(normal)            if normal            else "None")
        print("Tumor samples:",       ", ".join(tumor)             if tumor             else "None")
        print("Unspecified samples:", ", ".join(unspecified)       if unspecified       else "None")
        print("Manually excluded:",   ", ".join(manually_excluded) if manually_excluded else "None")
        print("Excluded (non-human):", ", ".join(excluded_non_human) if excluded_non_human else "None")
        print("Excluded (non-scRNA):", ", ".join(excluded_non_scrna) if excluded_non_scrna else "None")

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
        for dirpath, dirnames, filenames in os.walk(root):
            lower        = [f.lower() for f in filenames]
            has_matrix   = any("matrix"   in f and (f.endswith(".mtx.gz") or f.endswith(".mtx"))   for f in lower)
            has_features = any(("features" in f or "genes" in f) and (f.endswith(".tsv.gz") or f.endswith(".tsv")) for f in lower)
            has_barcodes = any("barcodes" in f and (f.endswith(".tsv.gz") or f.endswith(".tsv"))   for f in lower)
            if has_matrix and has_features and has_barcodes:
                return dirpath
        return None

    def _rename_mtx_files(self, mtx_dir: str):
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

            self._extract_tarballs(gsm_dir)
            files = os.listdir(gsm_dir)
            adata = None

            mtx_dir = self._find_mtx_dir(gsm_dir)
            if mtx_dir is not None:
                self._rename_mtx_files(mtx_dir)
                try:
                    print(f"Reading MTX matrix for {gsm_id} (from {os.path.relpath(mtx_dir, gse_dir)})")
                    adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", cache=False)
                except Exception as exc:
                    print(f"  MTX read failed for {gsm_id}: {exc}")

            if adata is None:
                for f in files:
                    fl = f.lower()
                    if fl.endswith(".tar.gz") or fl.endswith(".tar"):
                        continue
                    if any(fl.endswith(ext) for ext in [".tsv", ".csv", ".txt", ".gz"]) and "matrix" in fl:
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
