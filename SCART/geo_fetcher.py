import GEOparse
import os
import scanpy as sc
import anndata as ad
import pandas as pd
import gzip
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ✅ NEW: Tabula reference info
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
    cancer_type : str or None, optional
        Manually specify the cancer type instead of auto-detecting it from
        GEO metadata.  Must be one of the keys in ``TABULA_FILES`` (e.g.
        ``"ovary_cancer"``, ``"lung_cancer"``).  When provided, auto-detection
        is skipped entirely and the supplied value is used for reference
        guidance.

        Accepted values
        ---------------
        Any key from ``TABULA_FILES``:
            bladder_cancer, blood_cancer, bone_marrow_cancer, breast_cancer,
            ear_cancer, eye_cancer, fat_cancer, heart_cancer, kidney_cancer,
            large_intestine_cancer, liver_cancer, lung_cancer,
            lymph_node_cancer, muscle_cancer, ovary_cancer, pancreas_cancer,
            prostate_cancer, salivary_gland_cancer, skin_cancer,
            small_intestine_cancer, spleen_cancer, stomach_cancer,
            testis_cancer, thymus_cancer, tongue_cancer, trachea_cancer,
            uterus_cancer, vasculature_cancer.

        Multiple types can be provided as a comma-separated string:
            ``"ovary_cancer, lung_cancer"``

        To print all valid values:
            ``from SCART.geo_fetcher import VALID_CANCER_TYPES``
            ``print(VALID_CANCER_TYPES)``

        If not provided (default: None), cancer type is inferred
        automatically from GEO metadata text.

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
            manual_annotation_col="cell_type",   # your obs column name
            min_genes=200,
            max_mt=40,
        )

        If not provided (default: None), PopV runs normally on the h5ad.

    Notes
    -----
    GEO ID inputs always run the full PopV pipeline regardless of
    manual_annotation_col — the parameter is silently ignored for GEO IDs.
    """

    def __init__(
        self,
        *inputs,
        min_genes: int = None,
        max_mt: float = None,
        cancer_type: str = None,
        manual_annotation_col: str = None,
    ):

        self.inputs    = list(inputs)
        self.base_dir  = "GSE_data"

        # ── QC parameters: None means "user did not set this → skip QC" ───
        self.min_genes = min_genes
        self.max_mt    = max_mt

        # ── Manual cancer type override ────────────────────────────────────
        self._user_cancer_type = None
        if cancer_type is not None:
            self._user_cancer_type = self._validate_cancer_type(cancer_type)

        # ── Manual annotation: only used when h5ad files are provided ──────
        self.manual_annotation_col = manual_annotation_col

        os.makedirs(self.base_dir, exist_ok=True)

        self.gse_ids      = []
        self.h5ad_inputs  = []

        for item in self.inputs:
            if isinstance(item, str) and item.lower().endswith(".h5ad"):
                self.h5ad_inputs.append(item)
            else:
                self.gse_ids.append(item)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_cancer_type(self, cancer_type: str) -> str:
        """
        Validate a user-supplied cancer type string.

        Accepts a single key or a comma-separated list of keys.  Each token
        is checked against TABULA_FILES.  A ValueError is raised if any token
        is unrecognised so the user gets immediate, actionable feedback.

        Returns the normalised (stripped) string unchanged.
        """
        tokens = [t.strip() for t in cancer_type.split(",")]
        invalid = [t for t in tokens if t not in TABULA_FILES]

        if invalid:
            raise ValueError(
                f"\ncancer_type contains unrecognised value(s): {invalid}\n\n"
                f"Valid cancer types are:\n"
                + "\n".join(f"  {k}" for k in VALID_CANCER_TYPES)
                + "\n\nPass them as a string, e.g.:\n"
                "  cancer_type='ovary_cancer'\n"
                "  cancer_type='ovary_cancer, lung_cancer'\n\n"
                "To see all valid values:\n"
                "  from SCART.geo_fetcher import VALID_CANCER_TYPES\n"
                "  print(VALID_CANCER_TYPES)"
            )

        return ", ".join(tokens)

    def _store_qc_params(self, adata):
        """
        Write QC thresholds into adata.uns['qc_params'] only when the user
        has explicitly provided at least one threshold.

        If neither min_genes nor max_mt was set, the key is removed (or
        never written) so that Module 3 knows to skip QC entirely.
        """
        if self.min_genes is None and self.max_mt is None:
            # Ensure the key does not exist (e.g. from a previous run)
            adata.uns.pop("qc_params", None)
            return

        adata.uns["qc_params"] = {
            "min_genes": self.min_genes,   # may still be None for one field
            "max_mt":    self.max_mt,
        }

    def _store_manual_annotation(self, adata, source_file: str):
        """
        Validate and store manual annotation metadata when the user has
        supplied manual_annotation_col.

        What this does
        --------------
        1. Checks the column exists in adata.obs — raises ValueError if not.
        2. Copies the column into 'popv_majority_vote_prediction' so that
           Module 3 (preprocessing.py) can find it without any code changes.
           If 'popv_majority_vote_prediction' already exists it is overwritten
           and a warning is printed.
        3. Stores adata.uns['manual_annotation_col'] = column name so any
           downstream module can know which original column was used.
        4. Stores adata.uns['skip_popv'] = True so Module 2
           (auto_run_popv) can detect and skip the PopV pipeline.
        5. Prints a summary of unique labels found in the column so the
           user can verify their annotation is being read correctly.

        Parameters
        ----------
        adata       : AnnData object loaded from the user-supplied h5ad.
        source_file : path string used only for error messages.
        """
        col = self.manual_annotation_col

        # ── 1. Validate column exists ────────────────────────────────────
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

        # ── 2. Copy into popv_majority_vote_prediction ───────────────────
        if "popv_majority_vote_prediction" in adata.obs.columns:
            print(
                f"  WARNING: 'popv_majority_vote_prediction' already exists "
                f"in adata.obs — overwriting with values from '{col}'."
            )
        adata.obs["popv_majority_vote_prediction"] = (
            adata.obs[col].astype(str)
        )

        # ── 3 & 4. Store metadata flags ──────────────────────────────────
        adata.uns["manual_annotation_col"] = col
        adata.uns["skip_popv"]             = True

        # ── 5. Print label summary for user verification ─────────────────
        unique_labels = sorted(adata.obs["popv_majority_vote_prediction"].unique())
        epithelial    = [l for l in unique_labels
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

    def _print_reference_guidance(self, cancer_type):

        print("\n========== REFERENCE GUIDANCE ==========")

        if self._user_cancer_type is not None:
            print("ℹ️  Cancer type was provided manually (auto-detection skipped).")

        if self.h5ad_inputs:
            if self.manual_annotation_col:
                print("👉 You provided your own h5ad file WITH manual annotations.")
                print("👉 PopV (Module 2) will be SKIPPED automatically.")
                print("👉 Proceed directly to Module 3 (preprocessing).")
            else:
                print("👉 You provided your own h5ad file.")
                print("👉 Please provide your own reference file for PopV.")
            return

        if cancer_type is None:
            print("❗ Cancer type not detected.")
            print("👉 Please provide your own reference file.")
            return

        cancers = [c.strip() for c in cancer_type.split(",")]

        for c in cancers:

            if c in TABULA_FILES:
                print(f"\n✅ Detected cancer type: {c}")
                print(f"👉 Download: {TABULA_FILES[c]}")
                print(f"👉 From: {TABULA_DOI_LINK}")
                print("👉 Use this in next module (PopV)")
            else:
                print(f"\n⚠️ Detected cancer type: {c}")
                print("❌ Not available in Tabula Sapiens")
                print("👉 Provide your own reference file")

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def run(self):

        normal       = []
        tumor        = []
        unspecified  = []
        annotation_info = {}
        cancer_type  = None

        tumor_adatas = []
        results      = {}

        for gse_id in self.gse_ids:

            n, t, u, ann, ct = self._process_gse(gse_id)

            normal.extend(n)
            tumor.extend(t)
            unspecified.extend(u)

            annotation_info.update(ann)

            # ── Use user-supplied cancer type if provided, else auto-detected
            if self._user_cancer_type is not None:
                ct = self._user_cancer_type

            if ct and cancer_type is None:
                cancer_type = ct

            adata = self._build_h5ad(
                gse_id,
                t,
                save_single=(len(self.gse_ids) == 1 and len(self.h5ad_inputs) == 0),
                cancer_type_override=self._user_cancer_type,
            )

            if adata is not None:
                tumor_adatas.append(adata)

            results[gse_id] = (n, t, u, ann, None, ct)

        for file in self.h5ad_inputs:

            print("\n========== Reading h5ad file ==========")

            adata = sc.read_h5ad(file)
            adata.obs_names_make_unique()
            adata.layers["counts"] = adata.X.copy()
            adata.raw = adata

            # ── Store user-supplied cancer type in h5ad if provided ───────
            if self._user_cancer_type is not None:
                adata.uns["cancer_type"] = self._user_cancer_type
                print(f"  cancer_type stored (user-supplied): {self._user_cancer_type}")

            # ── Handle manual annotation if provided ─────────────────────
            if self.manual_annotation_col is not None:
                print("\n  Manual annotation mode activated.")
                self._store_manual_annotation(adata, file)

            # Store QC params in user-supplied h5ad too (only if user set them)
            self._store_qc_params(adata)

            tumor_adatas.append(adata)

            results[file] = ([], [], [], {}, None, self._user_cancer_type)

        query_h5ad  = None
        total_inputs = len(self.gse_ids) + len(self.h5ad_inputs)

        if total_inputs == 1:

            if len(self.gse_ids) == 1:

                query_h5ad = f"{self.gse_ids[0]}_tumor.h5ad"

                results[self.gse_ids[0]] = (
                    normal, tumor, unspecified,
                    annotation_info, query_h5ad, cancer_type
                )

            elif len(self.h5ad_inputs) == 1:

                adata    = tumor_adatas[0]
                filename = "input_tumor.h5ad"

                # QC params already handled by _store_qc_params above
                adata.write(filename)

                print("\n========== h5ad created ==========")
                print(f"{filename} is created successfully")

                if self.manual_annotation_col is not None:
                    print(
                        f"Manual annotation stored  → "
                        f"col='{self.manual_annotation_col}', "
                        f"skip_popv=True"
                    )
                    print("Next step: run Module 3 (preprocessing) directly.")
                    print("           Module 2 (PopV) is not needed.")

                if self.min_genes is not None or self.max_mt is not None:
                    print(
                        f"QC params stored → "
                        f"min_genes={self.min_genes}, max_mt={self.max_mt}"
                    )
                else:
                    print(
                        "QC step disabled "
                        "(no min_genes / max_mt provided — will be skipped in Module 3)"
                    )

                query_h5ad = filename
                key        = self.h5ad_inputs[0]

                results[key] = ([], [], [], {}, query_h5ad, self._user_cancer_type)

        elif total_inputs > 1 and len(tumor_adatas) > 0:

            combined = ad.concat(tumor_adatas, join="outer")
            combined.obs_names_make_unique()
            combined.layers["counts"] = combined.X.copy()
            combined.raw = combined

            # ── Store user-supplied cancer type in combined h5ad ──────────
            if self._user_cancer_type is not None:
                combined.uns["cancer_type"] = self._user_cancer_type
                print(f"\n  cancer_type stored in combined h5ad (user-supplied): "
                      f"{self._user_cancer_type}")

            # ── Re-apply manual annotation to combined object ─────────────
            # ad.concat does not carry uns from individual objects, so we
            # re-derive popv_majority_vote_prediction from the column that
            # was already copied into each individual adata before concat.
            if self.manual_annotation_col is not None:
                if "popv_majority_vote_prediction" in combined.obs.columns:
                    combined.uns["manual_annotation_col"] = self.manual_annotation_col
                    combined.uns["skip_popv"]             = True
                    print(
                        "\n  Combined h5ad: manual annotation carried through "
                        f"from column '{self.manual_annotation_col}'."
                    )
                else:
                    print(
                        "\n  WARNING: 'popv_majority_vote_prediction' lost during "
                        "concat — manual annotation NOT stored in combined h5ad."
                    )

            # Store QC params in the combined h5ad (only if user set them)
            self._store_qc_params(combined)

            combined.write("combined_tumor.h5ad")

            print("\n========== h5ad created ==========")
            print("combined_tumor.h5ad is created successfully")

            if self.manual_annotation_col is not None:
                print(
                    f"Manual annotation stored  → "
                    f"col='{self.manual_annotation_col}', "
                    f"skip_popv=True"
                )
                print("Next step: run Module 3 (preprocessing) directly.")

            if self.min_genes is not None or self.max_mt is not None:
                print(
                    f"QC params stored → "
                    f"min_genes={self.min_genes}, max_mt={self.max_mt}"
                )
            else:
                print(
                    "QC step disabled "
                    "(no min_genes / max_mt provided — will be skipped in Module 3)"
                )

            query_h5ad = "combined_tumor.h5ad"

            for key in results:
                n, t, u, ann, _, ct = results[key]
                results[key]        = (n, t, u, ann, query_h5ad, ct)

        # ── Final cancer type: user-supplied always wins ──────────────────
        if self._user_cancer_type is not None:
            cancer_type = self._user_cancer_type

        self._print_reference_guidance(cancer_type)

        return normal, tumor, unspecified, annotation_info, query_h5ad, cancer_type, results

    # ------------------------------------------------------------------
    # GEO processing
    # ------------------------------------------------------------------

    def _process_gse(self, gse_id):

        gse_dir = os.path.join(self.base_dir, gse_id)
        os.makedirs(gse_dir, exist_ok=True)

        gse = GEOparse.get_GEO(geo=gse_id, destdir=gse_dir)
        gse.download_supplementary_files(gse_dir)

        normal      = []
        tumor       = []
        unspecified = []
        annotation_info = {}

        excluded_non_scrna  = []
        excluded_non_human  = []

        # ── Cancer type: user-supplied overrides auto-detection ───────────
        if self._user_cancer_type is not None:
            cancer_type = self._user_cancer_type
            print(f"\n  Cancer type (user-supplied): {cancer_type}")
        else:
            cancer_type = self._predict_cancer_type(gse)

        tumor_keywords  = [
            "tumor", "tumour", "cancer", "carcinoma",
            "adenocarcinoma", "malignant", "metastatic"
        ]
        normal_keywords = [
            "normal", "healthy", "control", "adjacent normal"
        ]

        for gsm_id, gsm in gse.gsms.items():

            text = " ".join(
                [str(v) for v in gsm.metadata.values()]
            ).lower()

            organism = " ".join(gsm.metadata.get("organism_ch1", [])).lower()
            if "homo sapiens" not in organism:
                excluded_non_human.append(gsm_id)
                continue

            library = " ".join(gsm.metadata.get("library_strategy", [])).lower()
            if not any(k in library for k in ["rna-seq", "scrna", "single cell"]):
                excluded_non_scrna.append(gsm_id)
                continue

            label = "unspecified"

            if any(k in text for k in tumor_keywords):
                tumor.append(gsm_id)
                label = "tumor"
            elif any(k in text for k in normal_keywords):
                normal.append(gsm_id)
                label = "normal"
            else:
                unspecified.append(gsm_id)

            annotation_info[gsm_id] = label

        print(f"\n========== SAMPLE SUMMARY: {gse_id} ==========")
        print(f"Cancer type: {cancer_type}")
        print("Normal samples:",      ", ".join(normal)              if normal              else "None")
        print("Tumor samples:",       ", ".join(tumor)               if tumor               else "None")
        print("Unspecified samples:", ", ".join(unspecified)         if unspecified         else "None")
        print("Excluded (non-human):", ", ".join(excluded_non_human) if excluded_non_human else "None")
        print("Excluded (non-scRNA):", ", ".join(excluded_non_scrna) if excluded_non_scrna else "None")

        return normal, tumor, unspecified, annotation_info, cancer_type

    # ------------------------------------------------------------------

    def _predict_cancer_type(self, gse):

        text = (
            gse.metadata.get("title",   [""])[0] +
            " " +
            gse.metadata.get("summary", [""])[0]
        ).lower()

        cancer_map = {
            "ovary": [
                "ovarian", "ovary", "ovarian cancer", "ovarian carcinoma",
                "high-grade serous", "hgsoc", "lgsoc", "clear cell ovarian",
                "endometrioid ovarian", "mucinous ovarian",
                "fallopian tube", "peritoneal carcinoma",
            ],
            "uterus": [
                "uterine", "endometrial", "uterus", "endometrium",
                "uterine carcinoma", "endometrial carcinoma",
                "uterine sarcoma", "leiomyosarcoma uterine",
                "uterine leiomyoma", "cervical", "cervix",
                "cervical cancer", "cervical carcinoma",
            ],
            "lung": [
                "lung", "pulmonary", "nsclc", "sclc",
                "non-small cell lung", "small cell lung",
                "lung adenocarcinoma", "luad",
                "lung squamous", "lusc",
                "mesothelioma", "pleural",
                "bronchial", "bronchioloalveolar",
            ],
            "kidney": [
                "renal", "kidney", "rcc",
                "renal cell carcinoma", "clear cell renal",
                "papillary renal", "chromophobe renal",
                "wilms tumor", "nephroblastoma",
                "oncocytoma",
            ],
            "liver": [
                "liver", "hepatocellular", "hcc",
                "hepatic", "hepatocellular carcinoma",
                "intrahepatic cholangiocarcinoma", "icc",
                "biliary", "bile duct", "cholangiocarcinoma",
                "gallbladder",
            ],
            "pancreas": [
                "pancreatic", "pancreas", "pdac",
                "pancreatic ductal adenocarcinoma",
                "pancreatic cancer", "pancreatic carcinoma",
                "islet cell", "neuroendocrine pancreatic",
                "pancreatic neuroendocrine", "pnet",
                "acinar cell",
            ],
            "prostate": [
                "prostate", "prostatic",
                "prostate cancer", "prostate adenocarcinoma",
                "castration-resistant", "crpc",
                "neuroendocrine prostate",
            ],
            "bladder": [
                "bladder", "urothelial",
                "bladder cancer", "bladder carcinoma",
                "transitional cell carcinoma", "tcc",
                "urothelial carcinoma", "upper tract urothelial",
                "urinary tract",
            ],
            "stomach": [
                "gastric", "stomach",
                "gastric cancer", "gastric carcinoma",
                "gastroesophageal", "gej",
                "signet ring", "diffuse gastric",
                "intestinal gastric",
            ],
            "small_intestine": [
                "small intestine", "small bowel",
                "duodenal", "jejunal", "ileal",
                "small intestinal carcinoma",
                "gastrointestinal stromal", "gist",
            ],
            "large_intestine": [
                "colon", "colorectal", "rectal", "rectum",
                "crc", "colorectal cancer", "colon cancer",
                "colonic", "cecal", "cecum",
                "microsatellite instability", "msi",
                "mismatch repair", "mmr",
            ],
            "skin": [
                "melanoma", "skin", "cutaneous",
                "squamous cell skin", "basal cell",
                "merkel cell", "dermal",
                "uveal melanoma", "acral melanoma",
                "cutaneous melanoma", "melanocytic",
            ],
            "brain": [
                "glioma", "brain", "glioblastoma", "gbm",
                "astrocytoma", "oligodendroglioma",
                "medulloblastoma", "ependymoma",
                "meningioma", "craniopharyngioma",
                "diffuse intrinsic pontine", "dipg",
                "central nervous system", "cns tumor",
                "neural", "neuro-oncology",
            ],
            "blood": [
                "leukemia", "pbmc", "peripheral blood mononuclear",
                "aml", "cml", "all", "cll", "myeloma",
                "acute myeloid leukemia", "chronic myeloid leukemia",
                "acute lymphoblastic leukemia", "chronic lymphocytic leukemia",
                "t-cell leukemia", "hairy cell leukemia",
                "myelodysplastic", "mds",
                "myeloproliferative", "polycythemia vera",
                "essential thrombocythemia",
                "hematopoietic", "haematopoietic",
            ],
            "lymph_node": [
                "lymphoma", "lymph node", "lymphatic",
                "diffuse large b-cell", "dlbcl",
                "follicular lymphoma", "mantle cell lymphoma",
                "burkitt lymphoma", "hodgkin", "hodgkin lymphoma",
                "non-hodgkin", "marginal zone lymphoma",
                "t-cell lymphoma", "anaplastic large cell",
                "primary mediastinal", "natural killer cell lymphoma",
            ],
            "bone_marrow": [
                "myeloma", "multiple myeloma", "bone marrow",
                "plasma cell", "plasmacytoma",
                "myelofibrosis", "aplastic anemia",
                "amyloidosis", "waldenström",
                "smoldering myeloma",
            ],
            "breast": [
                "breast", "mammary", "tnbc",
                "triple negative breast", "triple-negative breast",
                "her2", "her2-positive breast",
                "luminal a", "luminal b",
                "breast adenocarcinoma", "breast carcinoma",
                "invasive ductal carcinoma", "idc",
                "invasive lobular carcinoma", "ilc",
                "ductal carcinoma in situ", "dcis",
                "inflammatory breast",
            ],
            "heart": [
                "cardiac", "heart",
                "cardiac tumor", "cardiac sarcoma",
                "cardiac rhabdomyoma", "cardiac fibroma",
                "cardiac myxoma", "pericardial",
                "myocardial",
            ],
            "thyroid": [
                "thyroid", "thyroid cancer", "thyroid carcinoma",
                "papillary thyroid", "ptc",
                "follicular thyroid", "ftc",
                "medullary thyroid", "mtc",
                "anaplastic thyroid", "atc",
                "hurthle cell",
            ],
            "esophagus": [
                "esophageal", "esophagus", "oesophageal",
                "esophageal adenocarcinoma", "eac",
                "esophageal squamous", "escc",
                "barrett", "barrett's esophagus",
            ],
            "trachea": [
                "trachea", "tracheal",
                "tracheal carcinoma", "tracheal tumor",
                "airway tumor",
            ],
            "tongue": [
                "tongue", "lingual",
                "tongue cancer", "tongue carcinoma",
                "oral tongue squamous",
            ],
            "salivary_gland": [
                "salivary", "salivary gland",
                "parotid", "submandibular",
                "mucoepidermoid carcinoma", "adenoid cystic",
                "acinic cell",
            ],
            "muscle": [
                "sarcoma", "muscle", "rhabdomyosarcoma",
                "leiomyosarcoma", "synovial sarcoma",
                "osteosarcoma", "ewing sarcoma",
                "soft tissue sarcoma", "undifferentiated sarcoma",
                "alveolar soft part sarcoma",
                "epithelioid sarcoma",
                "skeletal muscle tumor",
            ],
            "eye": [
                "ocular", "eye", "uveal",
                "retinoblastoma", "uveal melanoma",
                "conjunctival", "lacrimal gland",
                "intraocular", "orbital tumor",
            ],
            "ear": [
                "ear", "acoustic",
                "vestibular schwannoma", "acoustic neuroma",
                "middle ear tumor", "glomus jugulare",
            ],
            "fat": [
                "liposarcoma", "adipose", "lipoma",
                "well-differentiated liposarcoma", "wdlps",
                "dedifferentiated liposarcoma", "ddlps",
                "myxoid liposarcoma", "pleomorphic liposarcoma",
            ],
            "vasculature": [
                "vascular", "angiosarcoma", "endothelial",
                "hemangioendothelioma", "hemangioblastoma",
                "kaposi sarcoma", "glomus tumor",
                "arteriovenous malformation",
            ],
            "thymus": [
                "thymoma", "thymus", "thymic",
                "thymic carcinoma", "thymic epithelial",
                "anterior mediastinal",
            ],
            "testis": [
                "testicular", "testis",
                "germ cell tumor", "seminoma",
                "non-seminoma", "teratoma",
                "yolk sac tumor", "choriocarcinoma testicular",
                "leydig cell tumor", "sertoli cell tumor",
            ],
            "spleen": [
                "splenic", "spleen",
                "splenic lymphoma", "splenic marginal zone",
                "splenic hemangioma",
            ],
        }

        detected = []
        for tissue, keywords in cancer_map.items():
            if any(k in text for k in keywords):
                detected.append(f"{tissue}_cancer")

        if len(detected) == 0:
            return None

        return ", ".join(sorted(set(detected)))

    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------

    def _build_h5ad(self, gse_id, tumor_samples, save_single=False,
                    cancer_type_override=None):

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

        # ── Store cancer type: user-supplied takes priority ───────────────
        if cancer_type_override is not None:
            combined.uns["cancer_type"] = cancer_type_override
            print(f"... stored cancer_type in h5ad (user-supplied): {cancer_type_override}")
        else:
            cancer_type = self._predict_cancer_type(
                GEOparse.get_GEO(geo=gse_id, destdir=self.base_dir)
            )
            if cancer_type is not None:
                combined.uns["cancer_type"] = cancer_type
                print(f"... stored cancer_type in h5ad: {cancer_type}")

        # ── Store QC params only when user explicitly provided them ────────
        self._store_qc_params(combined)

        if self.min_genes is not None or self.max_mt is not None:
            print(
                f"... stored qc_params in h5ad: "
                f"min_genes={self.min_genes}, max_mt={self.max_mt}"
            )
        else:
            print(
                "... qc_params not stored "
                "(QC disabled — Module 3 will skip QC filtering)"
            )

        if save_single:

            filename = f"{gse_id}_tumor.h5ad"
            combined.write(filename)

            print("\n========== h5ad created ==========")
            print(f"{filename} is created successfully")

        return combined
