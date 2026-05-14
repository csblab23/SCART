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

# ── Disease-specific keywords used for the "unspecified → tumor" rescue pass ──
# These cover haematological and other malignancies that rarely use the
# word "tumor" in their sample descriptions.
DISEASE_TUMOR_KEYWORDS = [
    # ── General malignancy ───────────────────────────────────────────────────
    "tumor", "tumour", "cancer", "carcinoma", "adenocarcinoma",
    "malignant", "malignancy", "metastatic", "metastasis",
    "neoplasm", "neoplastic", "CAR-T", "infusion", "pre-infusion", "post-infusion", "leukapheresis",

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

    # ── Breast cancer molecular subtypes & staging ───────────────────────
    "tnbc", "triple negative",
    "her2+", "her2-positive", "her2 positive",
    "er+", "er-positive", "er positive",
    "pr+", "pr-positive", "pr positive",
    "luminal a", "luminal b",
    "dcis", "invasive ductal", "invasive lobular",

    # ── TNM staging — any T/N/M notation indicates a cancer patient ──────
    # Matches t1, t2, t3, t4 followed by n and m notation e.g. t2n1m0
    "t1n", "t2n", "t3n", "t4n",
    "tnm stage",

    # ── Other clinical cancer terms ───────────────────────────────────────
    "relapsed", "refractory", "recurrent",
    "post-treatment", "post treatment",
    "residual disease", "pathologic complete response",
    "overall survival", "disease-free survival",
]


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers (used by SampleAnnotator._build_h5ad)
# ─────────────────────────────────────────────────────────────────────────────

def _read_10x_h5_via_h5py(file_path: str):
    """
    Fallback HDF5 reader using h5py directly.

    Covers two CellRanger layouts:

    CellRanger v3 ("matrix" group)
    ───────────────────────────────
        /matrix/barcodes          — 1-D bytes array
        /matrix/features/id       — gene IDs
        /matrix/features/name     — gene symbols
        /matrix/data / indices / indptr / shape

    CellRanger v2 (genome-named group, e.g. "/GRCh38")
    ────────────────────────────────────────────────────
        /<genome>/barcodes
        /<genome>/gene_ids  (or gene_names)
        /<genome>/gene_names
        /<genome>/data / indices / indptr / shape

    sc.read_10x_h5 raises an empty KeyError when the HDF5 file uses the v2
    genome-group layout (no "matrix" key at the root).  sc.read_hdf5 also
    fails because the schema doesn't match its expectations.  This function
    reads both layouts with h5py directly, so v2 files are recovered instead
    of silently skipped.

    Returns AnnData or None.
    """
    import h5py
    import scipy.sparse as sp

    def _decode(arr):
        """Decode a bytes array to a list of str."""
        return [x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in arr]

    try:
        with h5py.File(file_path, "r") as f:

            # ── Try v3 layout first ("matrix" group at root) ─────────────
            if "matrix" in f:
                g = f["matrix"]
                data    = g["data"][:]
                indices = g["indices"][:]
                indptr  = g["indptr"][:]
                shape   = tuple(g["shape"][:])   # (n_genes, n_barcodes)

                barcodes = _decode(g["barcodes"][:])

                feat = g["features"]
                gene_ids   = _decode(feat["id"][:])
                gene_names = _decode(feat["name"][:]) if "name" in feat \
                             else gene_ids

                # shape is (genes, barcodes); we want cells × genes
                X = sp.csr_matrix(
                    (data, indices, indptr), shape=shape
                ).T

            # ── Fall back to v2 genome-group layout ──────────────────────
            else:
                # Pick the first non-metadata top-level group that contains
                # a "data" dataset (the CSR values array).
                genome_key = next(
                    (k for k in f.keys()
                     if isinstance(f[k], h5py.Group)
                     and "data" in f[k]),
                    None
                )
                if genome_key is None:
                    return None

                g = f[genome_key]
                data    = g["data"][:]
                indices = g["indices"][:]
                indptr  = g["indptr"][:]
                shape   = tuple(g["shape"][:])

                barcodes   = _decode(g["barcodes"][:])
                gene_ids   = _decode(g["gene_ids"][:])   if "gene_ids"   in g \
                             else _decode(g["gene_names"][:])
                gene_names = _decode(g["gene_names"][:]) if "gene_names" in g \
                             else gene_ids

                X = sp.csr_matrix(
                    (data, indices, indptr), shape=shape
                ).T

        obs = pd.DataFrame(index=barcodes)
        var = pd.DataFrame(
            {"gene_ids": gene_ids, "gene_symbols": gene_names},
            index=gene_names,   # use symbols as primary index (matches sc default)
        )

        return ad.AnnData(X=X, obs=obs, var=var)

    except Exception as exc:
        print(f"    h5py fallback read failed: {exc}")
        return None


def _dedup_var_names(adata: ad.AnnData) -> ad.AnnData:
    """
    Return *adata* with guaranteed-unique var_names.
    Appends .1, .2, … to duplicate names (same strategy as R's make.unique).
    No-op when var_names are already unique.
    """
    if adata.var_names.is_unique:
        return adata
    seen: dict = {}
    new_idx = []
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
    """
    Concatenate a list of AnnData objects with join="outer", guaranteeing
    that the union of var_names is unique before ad.concat is called.

    Why the naive retry loop fails
    ───────────────────────────────
    Per-sample deduplication (appending .1, .2, … within each sample) is
    not sufficient.  If sample A and sample B both contain a duplicate gene
    "TBCE", they each independently produce "TBCE.1".  The *union* of their
    var_names then contains "TBCE.1" twice — a collision that ad.concat with
    join="outer" cannot resolve, raising InvalidIndexError.

    Strategy
    ─────────
    1. Per-sample dedup  — make each adata's var_names unique in isolation.
    2. Union dedup       — build the union of all var_names; if it still
                           contains duplicates, apply a single global suffix
                           pass so every name in the union is unique, then
                           remap each sample's var_names accordingly.
    3. concat            — now safe to call with join="outer".

    Returns the combined AnnData, or raises if concat itself fails for an
    unrelated reason.
    """
    # Step 1: per-sample dedup
    adatas = [_dedup_var_names(a) for a in adatas]

    # Step 2: check whether the union of all var_names is itself unique
    all_names = []
    for a in adatas:
        all_names.extend(a.var_names.tolist())

    # dict.fromkeys preserves insertion order and removes duplicates
    union = list(dict.fromkeys(all_names))

    if len(union) == len(all_names) or len(set(union)) == len(union):
        # Union is already unique — go straight to concat
        return ad.concat(adatas, join="outer")

    # Step 3: union still has collisions — build a globally-unique remapping
    print("  Post-dedup union var_names still contain duplicates; "
          "applying global remapping …")
    seen: dict = {}
    global_map: dict = {}   # original_name → globally-unique_name
    for name in all_names:
        if name not in global_map:
            if name not in seen:
                seen[name] = 0
                global_map[name] = name
            else:
                seen[name] += 1
                global_map[name] = f"{name}.{seen[name]}"

    # Remap var_names on a copy of each adata so we don't mutate the originals
    remapped = []
    for a in adatas:
        new_names = [global_map.get(v, v) for v in a.var_names]
        a = a.copy()
        a.var_names = new_names
        remapped.append(a)

    return ad.concat(remapped, join="outer")


# ─────────────────────────────────────────────────────────────────────────────


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

    def _classify_gsm(self, gsm):
        """
        Classify a single GSM as normal / tumor / unspecified.

        Two-pass strategy
        -----------------
        Pass 1 — per-sample fields (title, source_name, characteristics)
            These describe THIS sample specifically and are checked first.
            If a normal keyword is found here AND no disease keyword is
            present in the same fields, the sample is labelled **normal**
            immediately — series-level disease text cannot override it.

        Pass 2 — full metadata text (fallback)
            Used only when Pass 1 finds no normal signal.  The full text
            blob (all metadata fields joined) is scanned with the same
            keyword sets.  This catches datasets where the sample type is
            described only in the series summary or protocol fields, and
            preserves the original behaviour for tumor / unspecified
            classification.

        Classification order (first match wins)
        ----------------------------------------
        1. Per-sample fields contain a normal keyword AND no disease
           keyword → **normal**
        2. Full text contains a disease keyword → **tumor**
        3. Full text contains a normal keyword AND no disease keyword
           → **normal**
        4. Neither → **unspecified**

        Design note
        -----------
        Blood / PBMC / immune samples from cancer patients are NOT
        excluded here — for blood cancers (AML, CLL, lymphoma etc.)
        those ARE the tumour samples.  Users who wish to exclude
        specific GSM IDs can filter the returned lists before
        passing them downstream.

        Returns
        -------
        "normal" | "tumor" | "unspecified"
        """
        normal_keywords = [
            "normal", "healthy", "control", "adjacent normal",
            "non-tumor", "non-tumour", "non-cancer",
            "benign", "non-malignant",
        ]

        # ── Pass 1: per-sample fields only ────────────────────────────────
        per_sample_fields = ["title", "source_name_ch1", "characteristics_ch1"]
        per_sample_text = " ".join(
            " ".join(gsm.metadata.get(f, []))
            for f in per_sample_fields
        ).lower()

        has_normal_kw_ps  = any(k in per_sample_text for k in normal_keywords)
        has_disease_kw_ps = any(k in per_sample_text for k in DISEASE_TUMOR_KEYWORDS)

        # Normal signal in per-sample fields with no disease signal in
        # those same fields → label as normal regardless of series text.
        if has_normal_kw_ps and not has_disease_kw_ps:
            return "normal"

        # ── Pass 2: full metadata text ────────────────────────────────────
        full_text = " ".join(
            [str(v) for v in gsm.metadata.values()]
        ).lower()

        has_disease_kw_full = any(k in full_text for k in DISEASE_TUMOR_KEYWORDS)
        has_normal_kw_full  = any(k in full_text for k in normal_keywords)

        if has_disease_kw_full:
            return "tumor"

        if has_normal_kw_full:
            return "normal"

        return "unspecified"

    # ──────────────────────────────────────────────────────────────────────

    def _download_gse_level_suppl(self, gse_id: str, gse_dir: str):
        """
        Download GSE-level supplementary files that GEOparse misses.

        GEOparse.download_supplementary_files only fetches per-GSM files.
        Some datasets (e.g. GSE161529) deposit shared files at the series
        level — for example a single ``GSE161529_features.tsv.gz`` used by
        all samples.  This method fetches those files directly from the
        NCBI GEO FTP into *gse_dir* so that Tier 2.5 can find them.

        The FTP listing is parsed for filenames that start with the GSE ID
        (case-insensitive).  Files that already exist locally are skipped.
        Network errors are caught and reported as warnings — a missing
        GSE-level features file will still trigger the synthetic fallback
        in Tier 2.5, so the pipeline remains functional.
        """
        import urllib.request, urllib.error, re

        # GEO FTP path: series/GSE<nnn>nnn/GSE<id>/suppl/
        series_stub = "GSE" + str(int(gse_id[3:]) // 1000) + "nnn"
        ftp_base    = (
            f"https://ftp.ncbi.nlm.nih.gov/geo/series/"
            f"{series_stub}/{gse_id}/suppl/"
        )

        try:
            with urllib.request.urlopen(ftp_base, timeout=30) as resp:
                listing = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            print(f"  Warning: could not fetch GSE-level FTP listing for "
                  f"{gse_id}: {exc}")
            return

        # Extract filenames from the HTML/FTP directory listing.
        # NCBI uses href="filename" links; we grab anything starting with
        # the GSE accession (case-insensitive).
        pattern = re.compile(
            r'href="(' + re.escape(gse_id) + r'[^"]+)"',
            re.IGNORECASE,
        )
        gse_files = pattern.findall(listing)

        if not gse_files:
            return  # Nothing to download

        # Only download lightweight shared reference files — features/genes/
        # barcodes TSVs that are deposited once for the whole series.
        # Explicitly skip _RAW.tar bundles (huge, duplicate per-sample data)
        # and any other large archives.
        def _is_shared_ref(fname: str) -> bool:
            fl = fname.lower()
            if fl.endswith("_raw.tar") or fl.endswith(".tar") or fl.endswith(".tar.gz"):
                return False
            return any(kw in fl for kw in ("features", "genes", "barcodes", "cell_types", "metadata"))

        for fname in gse_files:
            if not _is_shared_ref(fname):
                continue  # Skip large archives and non-reference files

            dest = os.path.join(gse_dir, fname)
            if os.path.exists(dest):
                continue  # Already downloaded

            url = ftp_base + fname
            print(f"  Downloading GSE-level supplementary file: {fname}")
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as exc:
                print(f"  Warning: failed to download {fname}: {exc}")

    # ──────────────────────────────────────────────────────────────────────

    def _download_missing_gsm_suppl(self, gse, gse_dir: str):
        """
        Re-scan every GSM's supplementary_file_* metadata and download any
        files that GEOparse's download_supplementary_files missed.

        GEOparse silently skips supplementary files beyond the first one in
        some datasets (e.g. GSE161529 where each GSM has both a barcodes file
        and a matrix file registered under supplementary_file_1 /
        supplementary_file_2).  This method iterates over all registered URLs
        and fetches any that are not yet present on disk.

        Files are placed in the same Supp_<GSM>_* directory that GEOparse
        already created (or would create), mirroring GEOparse's layout so
        that the existing tier logic finds them without any changes.
        """
        import urllib.request

        for gsm_id, gsm in gse.gsms.items():

            # Collect all supplementary file URLs from metadata
            urls = []
            for key, val in gsm.metadata.items():
                if key.startswith("supplementary_file") and val:
                    urls.extend(val)

            if not urls:
                continue

            # Find the existing Supp_<GSM>_* directory for this sample
            supp_dirs = [
                d for d in os.listdir(gse_dir)
                if d.startswith(f"Supp_{gsm_id}")
                   and os.path.isdir(os.path.join(gse_dir, d))
            ]
            if not supp_dirs:
                continue  # Directory not created yet — GEOparse will handle it

            gsm_supp_dir = os.path.join(gse_dir, supp_dirs[0])

            for url in urls:
                url = url.strip()
                if not url or url == "NONE":
                    continue

                fname = url.split("/")[-1]
                dest  = os.path.join(gsm_supp_dir, fname)

                if os.path.exists(dest):
                    continue  # Already present

                print(f"  Downloading missing supplementary file: {gsm_id}/{fname}")
                try:
                    urllib.request.urlretrieve(url, dest)
                except Exception as exc:
                    print(f"  Warning: failed to download {fname} for {gsm_id}: {exc}")

    # ──────────────────────────────────────────────────────────────────────

    def _process_gse(self, gse_id):
        """
        Download a GEO series, classify each GSM, and return lists of
        normal / tumor / unspecified sample IDs.
        """
        gse_dir = os.path.join(self.base_dir, gse_id)
        os.makedirs(gse_dir, exist_ok=True)

        gse = GEOparse.get_GEO(geo=gse_id, destdir=gse_dir)

        # Fetch any GSE-level supplementary files (e.g. a shared features
        # file) that GEOparse's per-GSM downloader would otherwise miss.
        self._download_gse_level_suppl(gse_id, gse_dir)

        # Skip supplementary download if every GSM already has a local
        # directory (either GSM<id>/ or Supp_GSM<id>*/).  This prevents
        # redundant network calls on every re-run of the same GSE ID.
        def _supp_present(gsm_id: str) -> bool:
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

        # GEOparse sometimes only downloads supplementary_file_1 and silently
        # skips supplementary_file_2, _3, etc.  Re-scan all GSM metadata and
        # fetch any registered files that are still missing from disk.
        self._download_missing_gsm_suppl(gse, gse_dir)

        normal      = []
        tumor       = []
        unspecified = []
        annotation_info = {}

        excluded_non_scrna = []
        excluded_non_human = []

        for gsm_id, gsm in gse.gsms.items():

            # Filter: human only
            organism = " ".join(gsm.metadata.get("organism_ch1", [])).lower()
            if "homo sapiens" not in organism:
                excluded_non_human.append(gsm_id)
                continue

            # Filter: scRNA-seq only
            # library_strategy is checked first.  Many 10x Chromium datasets
            # deposited on GEO have library_strategy = "OTHER" rather than
            # "RNA-Seq", so we use a two-pass approach:
            #   Pass A — explicit scRNA keywords in the library_strategy field.
            #   Pass B — if library_strategy is "other" (or absent), scan the
            #            full metadata text for scRNA-seq evidence: CellRanger
            #            output filenames, 10x / droplet / scRNA keywords in
            #            the data-processing or title fields.
            library = " ".join(gsm.metadata.get("library_strategy", [])).lower()

            _SCRNA_KEYWORDS = ["rna-seq", "scrna", "single cell", "single-cell",
                                "singlenucleus", "single nucleus", "snrna"]

            pass_a = any(k in library for k in _SCRNA_KEYWORDS)

            if not pass_a:
                # Pass B: check full metadata for scRNA evidence
                full_meta = " ".join(str(v) for v in gsm.metadata.values()).lower()
                _SCRNA_EVIDENCE = [
                    # CellRanger output file patterns
                    "feature_bc_matrix", "filtered_feature", "raw_feature",
                    "gene_bc_matrices", "barcodes.tsv", "matrix.mtx",
                    # Technology / protocol keywords
                    "10x chromium", "10x genomics", "chromium controller",
                    "dropseq", "drop-seq", "indrop", "indrops",
                    "scrna-seq", "scrna seq", "sc rna-seq",
                    "single-cell rna", "single cell rna",
                    "snrna-seq", "snrna seq", "single-nucleus rna",
                    "cellranger", "cell ranger", "seurat",
                ]
                pass_b = any(k in full_meta for k in _SCRNA_EVIDENCE)

                if not pass_b:
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
        """
        Read a generic CSV/TSV expression matrix.

        IMPORTANT: .mtx / .mtx.gz files are NOT accepted here — they are
        binary Matrix Market format and must be handled by the MTX readers
        (Tiers 1 / 2 / 2.5).  Passing an MTX to this method will always
        fail gracefully and return None.

        Orientation detection
        ---------------------
        GEO expression matrices are deposited in two orientations:
          - genes × cells  (rows = genes, columns = cells) — most common
          - cells × genes  (rows = cells, columns = genes) — less common

        We detect orientation by checking the first column: if it looks like
        gene names (strings, not numbers) the matrix is genes × cells and we
        transpose so that rows become cells.  This avoids the old shape-based
        heuristic which was unreliable and, more importantly, avoids calling
        df.T on a large dense DataFrame (which doubles peak memory usage).

        Memory safety
        -------------
        Large dense expression matrices (e.g. 30k genes × 5k cells) easily
        exceed available memory when loaded as a full pandas DataFrame and
        then converted column-by-column with apply(pd.to_numeric).  Instead
        we:
          1. Read with index_col=0 so the gene/cell ID column is the index.
          2. Cast the entire numeric block in one pass with astype(float32).
          3. Convert directly to a scipy sparse matrix before building AnnData.
        """
        import scipy.sparse as sp

        # Guard: reject MTX files — they are not CSV/TSV
        if file_path.lower().endswith(".mtx") or file_path.lower().endswith(".mtx.gz"):
            return None

        # Guard: reject .h5 / .h5.gz files — handled by Tier 4 / 4.5
        fl = file_path.lower()
        if fl.endswith(".h5") or fl.endswith(".hdf5") or fl.endswith(".h5.gz"):
            return None

        try:
            opener = gzip.open(file_path, 'rt') if file_path.endswith(".gz") else open(file_path, 'r')
            with opener as f:
                # Sniff separator from first line
                first_line = f.readline()
                sep = "\t" if "\t" in first_line else ","

            # Read with the first column as the index
            if file_path.endswith(".gz"):
                with gzip.open(file_path, 'rt') as f:
                    df = pd.read_csv(f, sep=sep, index_col=0)
            else:
                df = pd.read_csv(file_path, sep=sep, index_col=0)

            if df.empty:
                return None

            # Orientation detection: if the index looks like gene names
            # (non-numeric strings) and the columns look like cell barcodes
            # or numeric IDs, the matrix is genes × cells → transpose.
            # We check whether the index values are predominantly non-numeric.
            def _index_is_strings(idx) -> bool:
                sample = list(idx[:20])
                numeric_count = 0
                for v in sample:
                    try:
                        float(v)
                        numeric_count += 1
                    except (ValueError, TypeError):
                        pass
                return numeric_count < len(sample) / 2

            index_is_gene_names = _index_is_strings(df.index)
            cols_are_gene_names = _index_is_strings(df.columns)

            if index_is_gene_names and not cols_are_gene_names:
                # genes × cells — transpose to cells × genes
                df = df.T
            elif not index_is_gene_names and cols_are_gene_names:
                # already cells × genes — no transpose needed
                pass
            elif index_is_gene_names and cols_are_gene_names:
                # Both look like strings; fall back to shape heuristic
                if df.shape[0] > df.shape[1]:
                    # More rows than columns → likely genes × cells
                    df = df.T
            # else: both numeric → assume cells × genes, leave as-is

            # Cast to float32 in one pass (avoids per-column apply overhead)
            try:
                numeric_block = df.values.astype("float32")
            except (ValueError, TypeError):
                # Some cells may contain non-numeric strings; coerce via pandas
                df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
                numeric_block = df.values.astype("float32")

            # Convert to sparse to save memory downstream
            X_sparse = sp.csr_matrix(numeric_block)

            obs = pd.DataFrame(index=df.index.astype(str))
            var = pd.DataFrame(index=df.columns.astype(str))

            return ad.AnnData(X=X_sparse, obs=obs, var=var)

        except Exception:
            return None

    # ──────────────────────────────────────────────────────────────────────

    def _extract_tarballs(self, gsm_dir: str):
        """
        Extract any .tar.gz / .tar archives present in *gsm_dir* into that
        same directory.

        GEO sometimes ships the 10x MTX triplet packaged inside a tarball
        (e.g. ``GSM4257051_G2_filtered_feature_bc_matrix.tar.gz``).  This
        method unpacks every such archive it finds so that the subsequent
        MTX-scanning logic can locate the individual files.

        Already-extracted files are detected by checking whether the archive
        member paths already exist on disk; if they do the archive is skipped
        to avoid redundant work on re-runs.

        Parameters
        ----------
        gsm_dir : str
            Directory that was just located for this GSM sample.
        """
        for fname in os.listdir(gsm_dir):
            # Accept both .tar.gz and plain .tar
            if not (fname.endswith(".tar.gz") or fname.endswith(".tar")):
                continue

            tar_path = os.path.join(gsm_dir, fname)

            try:
                with tarfile.open(tar_path, "r:*") as tf:
                    members = tf.getmembers()

                    # Skip if every member already exists on disk
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

    # ──────────────────────────────────────────────────────────────────────

    def _find_mtx_dir_canonical(self, root: str):
        """
        Tier 1 — fast path.

        Walk *root* recursively and return the first directory that already
        contains all three 10x MTX files with EXACT canonical .gz names:
            matrix.mtx.gz
            features.tsv.gz  (or genes.tsv.gz)
            barcodes.tsv.gz

        Only .gz files are accepted — sc.read_10x_mtx requires gzip
        compression and will fail on plain .mtx / .tsv files.
        Plain-named and prefix-named files are handled by Tier 2.
        Directories named _canonical_* are skipped here — they are
        managed exclusively by Tier 2.

        Returns
        -------
        str or None
        """
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip stale/incomplete _canonical_ staging dirs — Tier 2 owns them
            if os.path.basename(dirpath).startswith("_canonical_"):
                continue
            lower = set(f.lower() for f in filenames)
            has_matrix   = "matrix.mtx.gz"   in lower
            has_features = ("features.tsv.gz" in lower or "genes.tsv.gz" in lower)
            has_barcodes = "barcodes.tsv.gz"  in lower
            if has_matrix and has_features and has_barcodes:
                return dirpath

        return None

    # ──────────────────────────────────────────────────────────────────────

    def _find_and_stage_prefix_named_mtx(self, root: str, gsm_id: str):
        """
        Tier 2 — prefix-named MTX triplet handler.

        GEO datasets frequently ship 10x files with a sample-ID or cohort
        prefix instead of the canonical bare names, e.g.:

            CID3586_matrix.mtx          <- uncompressed, inside CID3586/ subdir
            CID3586_barcodes.tsv
            CID3586_features.tsv

        or with GSM prefix and gz compression:

            GSM4909278_B1-MH0033-matrix.mtx.gz
            GSM4909278_B1-MH0033-barcodes.tsv.gz
            GSM4909278_B1-MH0033-features.tsv.gz

        sc.read_10x_mtx requires EXACTLY canonical names AND .gz compression.
        This method:

          1. Walks the entire GSM directory tree collecting files whose names
             contain the role keywords with valid extensions (.mtx/.mtx.gz,
             .tsv/.tsv.gz).

          2. Groups them by shared prefix (everything before the role keyword).

          3. For each complete triplet, copies files into a canonical
             subdirectory (_canonical_<hash>/) as matrix.mtx.gz,
             features.tsv.gz, barcodes.tsv.gz — gzip-compressing on the fly
             if the source file is not already compressed.

        Using gzip copy (not rename) is safe on re-runs — if the canonical
        directory already has all 3 .gz files it is returned immediately.

        Returns
        -------
        str or None
        """
        import shutil, hashlib

        def _role(fname: str):
            fl = fname.lower()
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
            fl  = fname.lower()
            idx = fl.find(role)
            return fl[:idx]

        # ── canonical name: always .gz regardless of source compression ───
        _CANON = {
            "matrix":   "matrix.mtx.gz",
            "features": "features.tsv.gz",
            "barcodes": "barcodes.tsv.gz",
        }

        def _copy_as_gz(src_path: str, dst_path: str):
            """Copy src to dst as gzip, compressing on-the-fly if needed."""
            import shutil as _sh
            if src_path.endswith(".gz"):
                _sh.copy2(src_path, dst_path)
            else:
                with open(src_path, "rb") as f_in, \
                        gzip.open(dst_path, "wb") as f_out:
                    _sh.copyfileobj(f_in, f_out)

        groups: dict = {}
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                role = _role(fname)
                if role is None:
                    continue
                pre = _prefix(fname, role)
                key = os.path.join(dirpath, pre)
                if key not in groups:
                    groups[key] = {}
                if role not in groups[key]:
                    groups[key][role] = os.path.join(dirpath, fname)

        for key, roles in groups.items():
            if not ("matrix" in roles and "barcodes" in roles and "features" in roles):
                continue

            tag       = hashlib.md5(key.encode()).hexdigest()[:8]
            canon_dir = os.path.join(root, f"_canonical_{tag}")

            # Clean up any stale canonical dir that lacks .gz files
            # (left by a previous broken run that copied uncompressed files)
            if os.path.isdir(canon_dir):
                staged = set(os.listdir(canon_dir))
                if all(v in staged for v in _CANON.values()):
                    return canon_dir   # already complete and correct
                # Stale — wipe and rebuild
                import shutil as _rmsh
                _rmsh.rmtree(canon_dir)

            os.makedirs(canon_dir, exist_ok=True)
            for role, src_path in roles.items():
                dst_path = os.path.join(canon_dir, _CANON[role])
                if not os.path.exists(dst_path):
                    _copy_as_gz(src_path, dst_path)

            print(f"  Staged prefix-named MTX triplet for {gsm_id} → "
                  f"{os.path.relpath(canon_dir, root)}")
            return canon_dir

        return None

    # ──────────────────────────────────────────────────────────────────────

    def _find_and_stage_shared_barcodes_features(
        self, gsm_dir: str, gse_dir: str, gsm_id: str
    ):
        """
        Tier 2.5 — shared barcodes/features handler.

        Some GEO datasets (e.g. GSE161529) ship a single shared
        ``barcodes.tsv.gz`` and ``features.tsv.gz`` / ``genes.tsv.gz`` at
        the GSE level (or in a sibling directory), while each per-sample
        supplement only contains its own ``<prefix>-matrix.mtx.gz``.

        This method:

        1. Searches the GSM directory tree for a matrix file
           (``*matrix*.mtx.gz`` or ``*matrix*.mtx``).

        2. If found, searches upward from *gsm_dir* through *gse_dir* and
           all immediate siblings of *gsm_dir* for barcodes and features
           files — accepting both canonical bare names and any
           prefix-named variants.

        3. If a complete triplet is assembled across directories, all three
           files are staged (gzip-copied) into a canonical subdirectory
           inside *gsm_dir* and the path is returned.

        This handles the pattern where barcodes and features are shared
        across all samples in a series but each sample has its own matrix.

        Returns
        -------
        str or None
        """
        import shutil, hashlib

        _CANON = {
            "matrix":   "matrix.mtx.gz",
            "features": "features.tsv.gz",
            "barcodes": "barcodes.tsv.gz",
        }

        def _copy_as_gz(src_path: str, dst_path: str):
            import shutil as _sh
            if src_path.endswith(".gz"):
                _sh.copy2(src_path, dst_path)
            else:
                with open(src_path, "rb") as f_in, \
                        gzip.open(dst_path, "wb") as f_out:
                    _sh.copyfileobj(f_in, f_out)

        def _is_role(fname: str, role: str) -> bool:
            fl = fname.lower()
            if role == "matrix":
                return ("matrix" in fl and
                        (fl.endswith(".mtx.gz") or fl.endswith(".mtx")))
            if role == "barcodes":
                return ("barcodes" in fl and
                        (fl.endswith(".tsv.gz") or fl.endswith(".tsv")))
            if role == "features":
                return (("features" in fl or "genes" in fl) and
                        (fl.endswith(".tsv.gz") or fl.endswith(".tsv")))
            return False

        def _find_role_in_dir(directory: str, role: str):
            """Return first matching file path for *role* in *directory*."""
            if not os.path.isdir(directory):
                return None
            for fname in os.listdir(directory):
                if _is_role(fname, role):
                    return os.path.join(directory, fname)
            return None

        # ── Step 1: find matrix file inside the GSM dir tree ─────────────
        # When gsm_dir is a shared GSE-level directory (multiple samples'
        # files coexist), prefer a matrix file whose name contains the GSM ID.
        # Fall back to the first matrix found only when in a dedicated dir.
        matrix_path     = None
        matrix_path_any = None  # first matrix found regardless of GSM match
        gsm_id_lower    = gsm_id.lower()

        for dirpath, _, filenames in os.walk(gsm_dir):
            for fname in filenames:
                if not _is_role(fname, "matrix"):
                    continue
                full_path = os.path.join(dirpath, fname)
                if gsm_id_lower in fname.lower():
                    matrix_path = full_path
                    break
                if matrix_path_any is None:
                    matrix_path_any = full_path
            if matrix_path:
                break

        # Use GSM-matched path; fall back to any matrix only when gsm_dir is
        # a dedicated per-GSM directory (not a shared GSE-level dir).
        if matrix_path is None and matrix_path_any is not None:
            if os.path.normpath(gsm_dir) != os.path.normpath(gse_dir):
                matrix_path = matrix_path_any

        if matrix_path is None:
            return None  # No matrix at all — nothing to do

        # ── Step 2: search for barcodes & features in candidate dirs ──────
        # Search order:
        #   a) same directory as the matrix file
        #   b) gsm_dir itself (if matrix is in a subdirectory)
        #   c) gse_dir (shared at series level)
        #   d) all immediate subdirectories of gse_dir (sibling samples /
        #      shared data directories deposited alongside sample dirs)
        matrix_dir = os.path.dirname(matrix_path)
        candidate_dirs = []

        # Always try the matrix's own directory first
        candidate_dirs.append(matrix_dir)

        # Then the GSM root (if different from the matrix dir)
        if gsm_dir != matrix_dir:
            candidate_dirs.append(gsm_dir)

        # Then the GSE root and its immediate children
        candidate_dirs.append(gse_dir)
        try:
            for entry in os.listdir(gse_dir):
                entry_path = os.path.join(gse_dir, entry)
                if os.path.isdir(entry_path) and entry_path not in candidate_dirs:
                    candidate_dirs.append(entry_path)
        except OSError:
            pass

        barcodes_path = None
        features_path = None

        for cdir in candidate_dirs:
            if barcodes_path is None:
                barcodes_path = _find_role_in_dir(cdir, "barcodes")
            if features_path is None:
                features_path = _find_role_in_dir(cdir, "features")
            if barcodes_path and features_path:
                break

        # If features/genes file is missing entirely, we will generate a
        # synthetic placeholder after staging — see Step 3 below.
        if barcodes_path is None:
            return None  # No barcodes at all — cannot proceed

        # features_path may be None here; handled in Step 3.

        # ── Step 3: stage the complete triplet into a canonical dir ───────
        tag       = hashlib.md5(matrix_path.encode()).hexdigest()[:8]
        canon_dir = os.path.join(gsm_dir, f"_canonical_shared_{tag}")

        if os.path.isdir(canon_dir):
            staged = set(os.listdir(canon_dir))
            if all(v in staged for v in _CANON.values()):
                return canon_dir  # Already complete — reuse

        os.makedirs(canon_dir, exist_ok=True)

        _copy_as_gz(matrix_path,   os.path.join(canon_dir, _CANON["matrix"]))
        _copy_as_gz(barcodes_path, os.path.join(canon_dir, _CANON["barcodes"]))

        if features_path is not None:
            _copy_as_gz(features_path, os.path.join(canon_dir, _CANON["features"]))
            features_note = os.path.relpath(features_path, gse_dir)
        else:
            # No features file found anywhere — generate a synthetic one.
            # Row count is read from the MTX header (3rd token of the size line).
            import scipy.io as _sio
            import gzip as _gz
            n_genes = None
            try:
                with _gz.open(matrix_path, "rt") as _mf:
                    for _line in _mf:
                        if _line.startswith("%"):
                            continue
                        n_genes = int(_line.split()[0])
                        break
            except Exception:
                pass

            feat_dst = os.path.join(canon_dir, _CANON["features"])
            with _gz.open(feat_dst, "wt") as _ff:
                if n_genes is not None:
                    for _i in range(1, n_genes + 1):
                        _ff.write(f"Gene{_i}\tGene{_i}\tGene Expression\n")
                # If n_genes could not be determined, write an empty file;
                # _read_10x_manual will fail gracefully and return None.
            features_note = "(synthetic — not deposited in GEO)"

        print(
            f"  Staged MTX triplet for {gsm_id} → "
            f"{os.path.relpath(canon_dir, gse_dir)}"
            f"\n    matrix   : {os.path.relpath(matrix_path, gse_dir)}"
            f"\n    barcodes : {os.path.relpath(barcodes_path, gse_dir)}"
            f"\n    features : {features_note}"
        )
        return canon_dir

    # ──────────────────────────────────────────────────────────────────────

    def _read_10x_manual(self, mtx_dir: str):
        """
        Manual 10x MTX reader — fallback when sc.read_10x_mtx fails.

        Handles CellRanger v2 (genes.tsv, 2-column) and v3 (features.tsv,
        3-column) formats, both compressed (.gz) and plain.  sc.read_10x_mtx
        calls sys.exit(1) on v2 files when var_names="gene_symbols" because
        it expects a third type column — this method avoids that entirely.

        Parameters
        ----------
        mtx_dir : str
            Directory containing matrix.mtx.gz, barcodes.tsv.gz, and
            features.tsv.gz or genes.tsv.gz.

        Returns
        -------
        AnnData or None
        """
        import scipy.io as sio2
        import pandas as pd

        files = set(os.listdir(mtx_dir))

        mtx_f = next((f for f in files if f == "matrix.mtx.gz"), None)
        bar_f = next((f for f in files if f == "barcodes.tsv.gz"), None)
        gen_f = next((f for f in files
                      if f in ("features.tsv.gz", "genes.tsv.gz")), None)

        if not all([mtx_f, bar_f, gen_f]):
            return None

        try:
            with gzip.open(os.path.join(mtx_dir, mtx_f)) as f:
                X = sio2.mmread(f).T.tocsr()

            with gzip.open(os.path.join(mtx_dir, bar_f), "rt") as f:
                barcodes = [l.strip() for l in f if l.strip()]

            with gzip.open(os.path.join(mtx_dir, gen_f), "rt") as f:
                lines = [l.strip().split("\t") for l in f if l.strip()]

            # Support 1-col, 2-col (v2), and 3-col (v3) features files
            gene_ids   = [l[0] for l in lines]
            gene_names = [l[1] if len(l) > 1 else l[0] for l in lines]

            # Make var index unique — duplicate gene symbols occur in
            # CellRanger v2 genes.tsv files and cause ad.concat to crash.
            # Strategy: use gene_ids (Ensembl) as the primary index since
            # those are unique; fall back to disambiguating gene_names if
            # gene_ids are also duplicated (rare but possible).
            if len(set(gene_ids)) == len(gene_ids):
                var_index = gene_ids
            else:
                # Disambiguate by appending a counter to duplicates
                seen = {}
                var_index = []
                for gid in gene_ids:
                    if gid in seen:
                        seen[gid] += 1
                        var_index.append(f"{gid}.{seen[gid]}")
                    else:
                        seen[gid] = 0
                        var_index.append(gid)

            var = pd.DataFrame(
                {"gene_ids": gene_ids, "gene_symbols": gene_names},
                index=var_index
            )
            obs = pd.DataFrame(index=barcodes)

            return ad.AnnData(X=X, obs=obs, var=var)

        except Exception as exc:
            print(f"    Manual MTX read error: {exc}")
            return None

    # ──────────────────────────────────────────────────────────────────────

    def _read_h5_gz(self, file_path: str):
        """
        Tier 4.5 — gzip-compressed HDF5 handler.

        GEO occasionally deposits CellRanger HDF5 files with an extra .gz
        layer (e.g. *_raw_gene_bc_matrices_h5.h5.gz).  sc.read_10x_h5 cannot
        read these directly; this method decompresses to a temp file first,
        then delegates to sc.read_10x_h5 / sc.read_hdf5.

        Parameters
        ----------
        file_path : str
            Path to the .h5.gz file on disk.

        Returns
        -------
        AnnData or None
        """
        import tempfile, shutil

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                tmp_path = tmp.name

            with gzip.open(file_path, "rb") as f_in, \
                 open(tmp_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

            try:
                adata = sc.read_10x_h5(tmp_path)
            except Exception:
                try:
                    adata = sc.read_hdf5(tmp_path)
                except Exception as exc:
                    print(f"    H5.gz read failed: {exc}")
                    adata = None

            # Make var_names unique immediately — sc.read_10x_h5 uses gene
            # symbols as the index by default and duplicate symbols (e.g.
            # "TBCE", "MATR3") are common, causing ad.concat to fail later.
            if adata is not None:
                adata = _dedup_var_names(adata)

            return adata

        except Exception as exc:
            print(f"    H5.gz decompress failed: {exc}")
            return None
        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

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
                # Priority 1: dedicated per-GSM supplement directory
                supp_dirs = [
                    d for d in os.listdir(gse_dir)
                    if d.startswith(f"Supp_{gsm_id}")
                       and os.path.isdir(os.path.join(gse_dir, d))
                ]
                if len(supp_dirs) > 0:
                    gsm_dir = os.path.join(gse_dir, supp_dirs[0])
                else:
                    # Priority 2: GSE-level layout — some datasets deposit all
                    # sample files in a single shared supplement directory
                    # (e.g. Supp_GSE161529_*/) rather than per-sample dirs.
                    # Use gse_dir as the search root; Tier 2.5 will locate the
                    # per-GSM matrix by scanning for filenames that contain the
                    # GSM ID, and will pair it with shared barcodes/features.
                    gse_supp_dirs = [
                        os.path.join(gse_dir, d)
                        for d in os.listdir(gse_dir)
                        if os.path.isdir(os.path.join(gse_dir, d))
                           and not d.startswith("Supp_GSM")   # exclude other GSMs' dirs
                    ]
                    # Check whether any GSE-level dir contains a file for this GSM
                    found_gse_dir = None
                    for candidate in [gse_dir] + gse_supp_dirs:
                        try:
                            files_here = os.listdir(candidate)
                        except OSError:
                            continue
                        if any(gsm_id.lower() in f.lower() for f in files_here):
                            found_gse_dir = candidate
                            break
                    if found_gse_dir is not None:
                        gsm_dir = found_gse_dir
                    else:
                        continue

            # ── Step 1: extract any tarballs present in gsm_dir ───────────
            self._extract_tarballs(gsm_dir)

            adata = None

            # ── Tier 1: canonical layout ───────────────────────────────────
            # Fast path — find a directory that already has exact canonical
            # filenames (matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz).
            mtx_dir = self._find_mtx_dir_canonical(gsm_dir)

            if mtx_dir is not None:
                try:
                    print(f"Reading MTX matrix for {gsm_id} "
                          f"(from {os.path.relpath(mtx_dir, gse_dir)})")
                    adata = sc.read_10x_mtx(
                        mtx_dir, var_names="gene_symbols", cache=False
                    )
                except (SystemExit, Exception) as exc:
                    print(f"  sc.read_10x_mtx failed ({type(exc).__name__}) "
                          f"— trying manual reader for {gsm_id}")
                    adata = self._read_10x_manual(mtx_dir)
                    if adata is None:
                        print(f"  Manual read also failed for {gsm_id}")

            # ── Tier 2: prefix-named layout ────────────────────────────────
            # Handles files like CID3586_matrix.mtx.gz (GSE176078) or
            # GSM4909278_B1-MH0033-matrix.mtx.gz (GSE161529) by copying
            # them into a canonical temp subdirectory and reading from there.
            if adata is None:
                staged_dir = self._find_and_stage_prefix_named_mtx(gsm_dir, gsm_id)
                if staged_dir is not None:
                    try:
                        print(f"Reading prefix-named MTX for {gsm_id} "
                              f"(staged at {os.path.relpath(staged_dir, gse_dir)})")
                        adata = sc.read_10x_mtx(
                            staged_dir, var_names="gene_symbols", cache=False
                        )
                    except (SystemExit, Exception) as exc:
                        print(f"  sc.read_10x_mtx failed ({type(exc).__name__}) "
                              f"— trying manual reader for {gsm_id}")
                        adata = self._read_10x_manual(staged_dir)
                        if adata is None:
                            print(f"  Manual read also failed for {gsm_id}")

            # ── Tier 2.5: shared barcodes/features layout ──────────────────
            # Handles GEO datasets where barcodes.tsv.gz and features.tsv.gz
            # are deposited once at the GSE level (shared across all samples)
            # while each per-sample supplement contains only its matrix file.
            # The three files may come from different directories; this tier
            # assembles them into a canonical staging directory.
            if adata is None:
                staged_dir = self._find_and_stage_shared_barcodes_features(
                    gsm_dir, gse_dir, gsm_id
                )
                if staged_dir is not None:
                    try:
                        print(f"Reading shared-barcodes MTX for {gsm_id} "
                              f"(staged at {os.path.relpath(staged_dir, gse_dir)})")
                        adata = sc.read_10x_mtx(
                            staged_dir, var_names="gene_symbols", cache=False
                        )
                    except (SystemExit, Exception) as exc:
                        print(f"  sc.read_10x_mtx failed ({type(exc).__name__}) "
                              f"— trying manual reader for {gsm_id}")
                        adata = self._read_10x_manual(staged_dir)
                        if adata is None:
                            print(f"  Manual read also failed for {gsm_id}")

            # ── Tier 3: generic CSV/TSV matrix ─────────────────────────────
            # Last resort for non-10x formats (CSV/TSV expression tables).
            # MTX files are explicitly excluded — they must be handled above.
            if adata is None:
                files = os.listdir(gsm_dir)
                for f in files:
                    fl = f.lower()
                    if fl.endswith(".tar.gz") or fl.endswith(".tar"):
                        continue
                    # Skip MTX files — they are binary and not CSV/TSV
                    if fl.endswith(".mtx") or fl.endswith(".mtx.gz"):
                        continue
                    if (any(fl.endswith(ext) for ext in [".tsv", ".csv", ".txt", ".gz"])
                            and ("matrix" in fl or "counts" in fl or "count" in fl)):
                        file_path = os.path.join(gsm_dir, f)
                        print(f"Reading generic matrix for {gsm_id}: {f}")
                        adata = self._read_generic_matrix(file_path)
                        if adata is not None:
                            break

            # ── Tier 4: HDF5 / CellRanger .h5 ────────────────────────────
            # CellRanger outputs filtered_feature_bc_matrix.h5 or
            # raw_feature_bc_matrix.h5 — readable with sc.read_10x_h5().
            # Also handles generic .h5 / .hdf5 files via sc.read_hdf5().
            #
            # Three-attempt strategy
            # ──────────────────────
            # Attempt 1: sc.read_10x_h5  — works for CellRanger v3 files
            #            (root group "matrix").  Raises an empty KeyError on
            #            v2 files (root group is genome name, e.g. "GRCh38").
            # Attempt 2: sc.read_hdf5    — generic HDF5; rarely succeeds on
            #            CellRanger files but worth trying before h5py.
            # Attempt 3: _read_10x_h5_via_h5py — direct h5py reader that
            #            handles both v2 and v3 layouts explicitly.  This
            #            recovers the files that caused "H5 read failed: "
            #            (empty error message) in the original code.
            if adata is None:
                files = os.listdir(gsm_dir)
                for f in sorted(files):   # sorted: prefer filtered over raw
                    fl = f.lower()
                    if fl.endswith(".h5") or fl.endswith(".hdf5"):
                        file_path = os.path.join(gsm_dir, f)
                        print(f"Reading H5 file for {gsm_id}: {f}")

                        # Attempt 1: scanpy CellRanger v3 reader
                        try:
                            adata = sc.read_10x_h5(file_path)
                        except Exception:
                            adata = None

                        # Attempt 2: scanpy generic HDF5 reader
                        if adata is None:
                            try:
                                adata = sc.read_hdf5(file_path)
                            except Exception:
                                adata = None

                        # Attempt 3: h5py fallback (handles v2 genome-group layout)
                        if adata is None:
                            print(f"  sc readers failed — trying h5py fallback for {gsm_id}")
                            adata = _read_10x_h5_via_h5py(file_path)
                            if adata is None:
                                print(f"  H5 read failed for {gsm_id} (all methods exhausted)")

                        # Deduplicate var_names immediately after any successful read
                        if adata is not None:
                            adata = _dedup_var_names(adata)
                            break

            # ── Tier 4.5: gzip-compressed HDF5 (.h5.gz) ──────────────────
            # GEO sometimes wraps CellRanger .h5 output in an extra gzip
            # layer (e.g. *_raw_gene_bc_matrices_h5.h5.gz).  The Tier 4
            # scanner above only matches bare .h5 / .hdf5 — this tier
            # catches the .h5.gz variant by decompressing to a temp file
            # first, then reading with sc.read_10x_h5 / sc.read_hdf5.
            if adata is None:
                files = os.listdir(gsm_dir)
                for f in sorted(files):
                    fl = f.lower()
                    if fl.endswith(".h5.gz"):
                        file_path = os.path.join(gsm_dir, f)
                        print(f"Reading H5.gz file for {gsm_id}: {f}")
                        adata = self._read_h5_gz(file_path)
                        if adata is not None:
                            break

            # ── Tier 5: Loom ──────────────────────────────────────────────
            # Loom is a HDF5-based format used by some pipelines (velocyto,
            # STARsolo).  sc.read_loom() handles it natively.
            if adata is None:
                files = os.listdir(gsm_dir)
                for f in files:
                    if f.lower().endswith(".loom"):
                        file_path = os.path.join(gsm_dir, f)
                        print(f"Reading Loom file for {gsm_id}: {f}")
                        try:
                            adata = sc.read_loom(file_path)
                        except Exception as exc:
                            print(f"  Loom read failed for {gsm_id}: {exc}")
                            adata = None
                        if adata is not None:
                            break

            # ── Tier 6: H5AD ─────────────────────────────────────────────
            # Some GEO deposits provide pre-built AnnData .h5ad files.
            if adata is None:
                files = os.listdir(gsm_dir)
                for f in files:
                    if f.lower().endswith(".h5ad"):
                        file_path = os.path.join(gsm_dir, f)
                        print(f"Reading H5AD file for {gsm_id}: {f}")
                        try:
                            adata = sc.read_h5ad(file_path)
                        except Exception as exc:
                            print(f"  H5AD read failed for {gsm_id}: {exc}")
                            adata = None
                        if adata is not None:
                            break

            if adata is None:
                print(f"Skipping {gsm_id} (no valid expression matrix found)")
                continue

            adata.obs["gsm_id"] = gsm_id
            adata.obs["gse_id"] = gse_id
            adata.layers["counts"] = adata.X.copy()
            adata.raw = adata
            adata.obs_names_make_unique()

            # Deduplicate var_names proactively — duplicate gene symbols
            # (common in CellRanger h5 and some CSV matrices) cause
            # ad.concat to raise InvalidIndexError even with join="outer".
            # _dedup_var_names is a no-op when var_names are already unique.
            adata = _dedup_var_names(adata)

            adatas.append(adata)

        if len(adatas) == 0:
            return None

        # ── Safe concat: guarantees the union var index is unique ──────────
        # The old try/except/retry pattern failed when two samples each
        # independently renamed a duplicate gene to the same suffix (e.g.
        # both produced "TBCE.1"), causing a collision in the outer-join union.
        # _safe_concat handles this with a global remapping pass if needed.
        try:
            combined = _safe_concat(adatas)
        except Exception as exc:
            print(f"  Warning: concat failed even after global dedup: {exc}")
            print("  Skipping this GSE — no h5ad will be written.")
            return None

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
