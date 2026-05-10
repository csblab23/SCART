import GEOparse
import os
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

# ── Disease-specific keywords used for the "unspecified → tumor" rescue pass ──
# These cover haematological and other malignancies that rarely use the
# word "tumor" in their sample descriptions.
DISEASE_TUMOR_KEYWORDS = [
    # ── General malignancy ───────────────────────────────────────────────────
    "tumor", "tumour", "cancer", "carcinoma", "adenocarcinoma",
    "malignant", "malignancy", "metastatic", "metastasis",
    "neoplasm", "neoplastic",

    # ── Haematological ───────────────────────────────────────────────────────
    "leukemia", "leukaemia", "lymphoma", "myeloma",
    "aml", "cml", "all", "cll", "mds",
    "acute myeloid", "chronic myeloid",
    "acute lymphoblastic", "chronic lymphocytic",
    "acute lymphocytic",
    "t-cell leukemia", "b-cell leukemia",
    "hairy cell leukemia", "large granular lymphocyte",
    "myelodysplastic", "myeloproliferative",
    "polycythemia vera", "essential thrombocythemia",
    "myelofibrosis",

    # ── Lymphoid ────────────────────────────────────────────────────────────
    "dlbcl", "follicular lymphoma", "mantle cell lymphoma",
    "burkitt lymphoma", "hodgkin", "non-hodgkin",
    "marginal zone lymphoma", "anaplastic large cell",
    "primary mediastinal b-cell",

    # ── Plasma cell ─────────────────────────────────────────────────────────
    "multiple myeloma", "plasma cell dyscrasia",
    "plasmacytoma", "waldenström",
    "smoldering myeloma", "amyloidosis",

    # ── Solid-tumour aliases missed by primary keywords ──────────────────────
    "hgsoc", "lgsoc", "pdac", "nsclc", "sclc",
    "gbm", "glioblastoma", "glioma", "astrocytoma",
    "melanoma", "sarcoma", "blastoma",
    "hepatocellular", "cholangiocarcinoma",
    "seminoma", "teratoma",
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
            cancer_type="my_custom_cancer"            # free-text, no Tabula ref

        To see all Tabula Sapiens keys::

            from SCART.geo_fetcher import VALID_CANCER_TYPES
            print(VALID_CANCER_TYPES)

    min_genes : int or None, optional
        Minimum number of genes detected per cell used for QC filtering in
        Module 3 (preprocessing).  The value is stored in
        ``adata.uns['qc_params']`` of every h5ad written by this module so
        that Module 3 can read it automatically.
        If not provided (default: None), the QC step is skipped in Module 3.

    max_mt : float or None, optional
        Maximum mitochondrial gene percentage per cell used for QC filtering
        in Module 3.  Stored alongside ``min_genes`` in
        ``adata.uns['qc_params']``.
        If not provided (default: None), the QC step is skipped in Module 3.

    manual_annotation_col : str or None, optional
        ONLY relevant when providing your own .h5ad file (not a GEO ID).

        Name of the column in adata.obs that contains your cell-type
        annotations.  When provided, Module 2 (PopV) is skipped entirely
        and your annotations are used directly in Module 3 (preprocessing).

        Requirements for the annotation column
        ---------------------------------------
        - Must exist in adata.obs of every h5ad file you pass.
        - Must contain string cell-type labels for every cell.
        - The column that identifies epithelial cells used by Module 3
          must contain labels ending with the phrase "epithelial cell"
          (case-insensitive).  Examples of valid epithelial labels:
              "epithelial cell"
              "glandular epithelial cell"
              "ovarian surface epithelial cell"
          Non-epithelial cells (all other labels) are used as the
          comparison "rest" group in the DEG step.
        - The column value is copied into a new column called
          "popv_majority_vote_prediction" so that Module 3 can find it
          without any changes to the downstream pipeline.

        What is stored in the h5ad
        --------------------------
        adata.uns['manual_annotation_col'] = <your column name>
            Tells downstream modules that PopV was skipped and which
            obs column holds the cell-type labels.
        adata.uns['skip_popv'] = True
            Explicit flag so Module 2 auto_run_popv() can detect and
            skip the PopV pipeline automatically.

        Usage example
        -------------
        annotator = SampleAnnotator(
            "my_data.h5ad",
            cancer_type="blood_cancer",
            manual_annotation_col="cell_type",
            min_genes=200,
            max_mt=40,
        )

        If not provided (default: None), PopV runs normally on the h5ad.

    Notes
    -----
    GEO ID inputs always run the full PopV pipeline regardless of
    manual_annotation_col — the parameter is silently ignored for GEO IDs.

    Sample classification logic
    ---------------------------
    Each GSM is first checked against normal/control keywords; if found it
    is labelled **normal**.  Next it is checked against a broad set of
    tumour/malignancy keywords (including haematological disease terms such
    as "aml", "leukemia", "myeloma", etc.); if found it is labelled
    **tumor**.  A sample reaches **unspecified** only when *neither* group
    of keywords is detected.

    This two-step approach means that blood-cancer datasets whose GSM
    descriptions use disease names instead of the word "tumor" (e.g.
    "AML patient", "CLL cells") are correctly classified as tumor rather
    than being silently dropped into the unspecified bucket.
    """

    def __init__(
        self,
        *inputs,
        cancer_type: str,
        min_genes: int = None,
        max_mt: float = None,
        manual_annotation_col: str = None,
    ):

        self.inputs   = list(inputs)
        self.base_dir = "GSE_data"

        # ── QC parameters ──────────────────────────────────────────────────
        self.min_genes = min_genes
        self.max_mt    = max_mt

        # ── Cancer type (required) ─────────────────────────────────────────
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

        # Parse and categorise each token as tabula or unknown
        self._user_cancer_type, self._tabula_types, self._unknown_types = (
            self._parse_cancer_type(cancer_type)
        )

        # ── Manual annotation ──────────────────────────────────────────────
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
        """
        Parse a (possibly comma-separated) cancer_type string.

        Each token is checked against TABULA_FILES:
          - Known tokens  → will get a Tabula Sapiens reference recommendation.
          - Unknown tokens → user will be told to supply their own reference.

        Returns
        -------
        normalised_str  : str   — joined, stripped version of all tokens
        tabula_types    : list  — tokens that exist in TABULA_FILES
        unknown_types   : list  — tokens not in TABULA_FILES
        """
        tokens = [t.strip() for t in cancer_type.split(",") if t.strip()]

        tabula_types  = [t for t in tokens if t in TABULA_FILES]
        unknown_types = [t for t in tokens if t not in TABULA_FILES]

        normalised = ", ".join(tokens)
        return normalised, tabula_types, unknown_types

    # ──────────────────────────────────────────────────────────────────────

    def _store_qc_params(self, adata):
        """
        Write QC thresholds into adata.uns['qc_params'] only when the user
        has explicitly provided at least one threshold.
        """
        if self.min_genes is None and self.max_mt is None:
            adata.uns.pop("qc_params", None)
            return

        adata.uns["qc_params"] = {
            "min_genes": self.min_genes,
            "max_mt":    self.max_mt,
        }

    # ──────────────────────────────────────────────────────────────────────

    def _store_manual_annotation(self, adata, source_file: str):
        """
        Validate and store manual annotation metadata when the user has
        supplied manual_annotation_col.
        """
        col = self.manual_annotation_col

        # 1. Validate column exists
        if col not in adata.obs.columns:
            available = list(adata.obs.columns)
            raise ValueError(
                f"\nmanual_annotation_col='{col}' not found in adata.obs "
                f"of '{source_file}'.\n"
                f"Available obs columns: {available}\n\n"
                "Please check the column name and try again.\n"
                "See the README section 'Manual annotation requirements' "
                "for full details."
            )

        # 2. Copy into popv_majority_vote_prediction
        if "popv_majority_vote_prediction" in adata.obs.columns:
            print(
                f"  WARNING: 'popv_majority_vote_prediction' already exists "
                f"in adata.obs — overwriting with values from '{col}'."
            )
        adata.obs["popv_majority_vote_prediction"] = (
            adata.obs[col].astype(str)
        )

        # 3 & 4. Store metadata flags
        adata.uns["manual_annotation_col"] = col
        adata.uns["skip_popv"]             = True

        # 5. Print label summary
        unique_labels  = sorted(adata.obs["popv_majority_vote_prediction"].unique())
        epithelial     = [l for l in unique_labels
                          if "epithelial cell" in l.lower()]
        non_epithelial = [l for l in unique_labels
                          if "epithelial cell" not in l.lower()]

        print(f"\n  Manual annotation column : '{col}'")
        print(f"  Copied to               : 'popv_majority_vote_prediction'")
        print(f"  Total unique labels     : {len(unique_labels)}")
        print(f"  Epithelial labels found : {epithelial if epithelial else 'NONE — check your label names!'}")
        print(f"  Non-epithelial labels   : {non_epithelial}")
        print( "  PopV will be SKIPPED for this file (adata.uns['skip_popv'] = True)")

        if not epithelial:
            print(
                "\n  ⚠ WARNING: No epithelial labels detected.\n"
                "  Module 3 identifies epithelial cells by matching labels that\n"
                "  END WITH 'epithelial cell' (case-insensitive), for example:\n"
                "    'epithelial cell'\n"
                "    'glandular epithelial cell'\n"
                "    'ovarian surface epithelial cell'\n"
                f"  Your labels in '{col}' do not match this pattern.\n"
                "  Module 3 will find 0 epithelial cells and may fail.\n"
                "  Please rename your epithelial label(s) to end with "
                "'epithelial cell'."
            )

    # ──────────────────────────────────────────────────────────────────────

    def _print_reference_guidance(self):
        """
        Print Tabula Sapiens reference guidance based on the user-supplied
        cancer type(s).  Unknown types are flagged with instructions to
        provide a custom reference.
        """
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
    # Public entry-point
    # ──────────────────────────────────────────────────────────────────────

    def run(self):

        normal      = []
        tumor       = []
        unspecified = []
        annotation_info = {}

        tumor_adatas = []
        results      = {}

        for gse_id in self.gse_ids:

            n, t, u, ann = self._process_gse(gse_id)

            normal.extend(n)
            tumor.extend(t)
            unspecified.extend(u)
            annotation_info.update(ann)

            adata = self._build_h5ad(
                gse_id,
                t,
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

            # Store cancer type
            adata.uns["cancer_type"] = self._user_cancer_type
            print(f"  cancer_type stored: {self._user_cancer_type}")

            # Handle manual annotation
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
                    print(
                        f"Manual annotation stored  → "
                        f"col='{self.manual_annotation_col}', skip_popv=True"
                    )
                    print("Next step: run Module 3 (preprocessing) directly.")

                if self.min_genes is not None or self.max_mt is not None:
                    print(f"QC params stored → min_genes={self.min_genes}, max_mt={self.max_mt}")
                else:
                    print("QC step disabled (no min_genes / max_mt provided — will be skipped in Module 3)")

                query_h5ad = filename
                key        = self.h5ad_inputs[0]
                results[key] = ([], [], [], {}, query_h5ad, self._user_cancer_type)

        elif total_inputs > 1 and len(tumor_adatas) > 0:

            combined = ad.concat(tumor_adatas, join="outer")
            combined.obs_names_make_unique()
            combined.layers["counts"] = combined.X.copy()
            combined.raw = combined

            combined.uns["cancer_type"] = self._user_cancer_type
            print(f"\n  cancer_type stored in combined h5ad: {self._user_cancer_type}")

            # Re-apply manual annotation after concat (uns is lost by concat)
            if self.manual_annotation_col is not None:
                if "popv_majority_vote_prediction" in combined.obs.columns:
                    combined.uns["manual_annotation_col"] = self.manual_annotation_col
                    combined.uns["skip_popv"]             = True
                    print(
                        f"\n  Combined h5ad: manual annotation carried through "
                        f"from column '{self.manual_annotation_col}'."
                    )
                else:
                    print(
                        "\n  WARNING: 'popv_majority_vote_prediction' lost during "
                        "concat — manual annotation NOT stored in combined h5ad."
                    )

            self._store_qc_params(combined)

            combined.write("combined_tumor.h5ad")

            print("\n========== h5ad created ==========")
            print("combined_tumor.h5ad is created successfully")

            if self.manual_annotation_col is not None:
                print(
                    f"Manual annotation stored  → "
                    f"col='{self.manual_annotation_col}', skip_popv=True"
                )
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

    def _classify_gsm(self, text: str):
        """
        Classify a single GSM based on its full metadata text.

        Classification order (first match wins)
        ----------------------------------------
        1. **normal**  – contains a normal/control/healthy keyword.
        2. **tumor**   – contains any tumour or disease-specific keyword
                         from DISEASE_TUMOR_KEYWORDS (covers both generic
                         terms like "tumor" and haematological terms like
                         "aml", "leukemia", "myeloma", etc.).
        3. **unspecified** – neither group matched.

        Returns
        -------
        "normal" | "tumor" | "unspecified"
        """
        normal_keywords = [
            "normal", "healthy", "control", "adjacent normal",
            "non-tumor", "non-tumour", "non-cancer",
            "benign", "non-malignant",
        ]

        if any(k in text for k in normal_keywords):
            return "normal"

        if any(k in text for k in DISEASE_TUMOR_KEYWORDS):
            return "tumor"

        return "unspecified"

    # ──────────────────────────────────────────────────────────────────────

    def _process_gse(self, gse_id):
        """
        Download a GEO series, classify each GSM, and return lists of
        normal / tumor / unspecified sample IDs.
        """
        gse_dir = os.path.join(self.base_dir, gse_id)
        os.makedirs(gse_dir, exist_ok=True)

        gse = GEOparse.get_GEO(geo=gse_id, destdir=gse_dir)
        gse.download_supplementary_files(gse_dir)

        normal      = []
        tumor       = []
        unspecified = []
        annotation_info = {}

        excluded_non_scrna = []
        excluded_non_human = []

        for gsm_id, gsm in gse.gsms.items():

            # Build a single lowercase text blob from all metadata fields
            text = " ".join(
                [str(v) for v in gsm.metadata.values()]
            ).lower()

            # Filter: human only
            organism = " ".join(gsm.metadata.get("organism_ch1", [])).lower()
            if "homo sapiens" not in organism:
                excluded_non_human.append(gsm_id)
                continue

            # Filter: scRNA-seq only
            library = " ".join(gsm.metadata.get("library_strategy", [])).lower()
            if not any(k in library for k in ["rna-seq", "scrna", "single cell"]):
                excluded_non_scrna.append(gsm_id)
                continue

            label = self._classify_gsm(text)

            if label == "normal":
                normal.append(gsm_id)
            elif label == "tumor":
                tumor.append(gsm_id)
            else:
                unspecified.append(gsm_id)

            annotation_info[gsm_id] = label

        print(f"\n========== SAMPLE SUMMARY: {gse_id} ==========")
        print(f"Cancer type (user-supplied): {self._user_cancer_type}")
        print("Normal samples:",       ", ".join(normal)              if normal              else "None")
        print("Tumor samples:",        ", ".join(tumor)               if tumor               else "None")
        print("Unspecified samples:",  ", ".join(unspecified)         if unspecified         else "None")
        print("Excluded (non-human):", ", ".join(excluded_non_human)  if excluded_non_human  else "None")
        print("Excluded (non-scRNA):", ", ".join(excluded_non_scrna)  if excluded_non_scrna  else "None")

        return normal, tumor, unspecified, annotation_info

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
                if len(supp_dirs) > 0:
                    gsm_dir = os.path.join(gse_dir, supp_dirs[0])
                else:
                    continue

            files = os.listdir(gsm_dir)

            matrix_file   = None
            features_file = None
            barcodes_file = None

            for f in files:
                if "matrix"   in f and f.endswith(".mtx.gz"):
                    matrix_file   = f
                elif "features" in f and f.endswith(".tsv.gz"):
                    features_file = f
                elif "barcodes" in f and f.endswith(".tsv.gz"):
                    barcodes_file = f

            adata = None

            if matrix_file and features_file and barcodes_file:

                try:
                    os.rename(os.path.join(gsm_dir, matrix_file),
                              os.path.join(gsm_dir, "matrix.mtx.gz"))
                except Exception:
                    pass
                try:
                    os.rename(os.path.join(gsm_dir, features_file),
                              os.path.join(gsm_dir, "features.tsv.gz"))
                except Exception:
                    pass
                try:
                    os.rename(os.path.join(gsm_dir, barcodes_file),
                              os.path.join(gsm_dir, "barcodes.tsv.gz"))
                except Exception:
                    pass

                try:
                    print(f"Reading MTX matrix for {gsm_id}")
                    adata = sc.read_10x_mtx(
                        gsm_dir,
                        var_names="gene_symbols",
                        cache=False
                    )
                except Exception:
                    print(f"Skipping {gsm_id} (MTX read failed)")

            if adata is None:
                for f in files:
                    if (
                        any(f.endswith(ext) for ext in [".tsv", ".csv", ".txt", ".gz"])
                        and "matrix" in f.lower()
                    ):
                        file_path = os.path.join(gsm_dir, f)
                        print(f"Reading generic matrix for {gsm_id}: {f}")
                        adata = self._read_generic_matrix(file_path)
                        if adata is not None:
                            break

            if adata is None:
                print(f"Skipping {gsm_id} (no valid expression matrix)")
                continue

            adata.obs["gsm_id"] = gsm_id
            adata.obs["gse_id"] = gse_id
            adata.layers["counts"] = adata.X.copy()
            adata.raw = adata
            adata.obs_names_make_unique()

            adatas.append(adata)

        if len(adatas) == 0:
            return None

        combined = ad.concat(adatas, join="outer")
        combined.obs_names_make_unique()

        print("... storing 'gsm_id' as categorical")
        print("... storing 'gse_id' as categorical")

        combined.obs["gsm_id"] = combined.obs["gsm_id"].astype("category")
        combined.obs["gse_id"] = combined.obs["gse_id"].astype("category")

        # Store user-supplied cancer type
        combined.uns["cancer_type"] = self._user_cancer_type
        print(f"... stored cancer_type in h5ad: {self._user_cancer_type}")

        # Store QC params
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
