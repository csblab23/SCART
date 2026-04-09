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


class SampleAnnotator:

    def __init__(self, *inputs):

        self.inputs = list(inputs)
        self.base_dir = "GSE_data"

        os.makedirs(self.base_dir, exist_ok=True)

        self.gse_ids = []
        self.h5ad_inputs = []

        for item in self.inputs:
            if isinstance(item, str) and item.lower().endswith(".h5ad"):
                self.h5ad_inputs.append(item)
            else:
                self.gse_ids.append(item)

    # ✅ NEW FUNCTION
    def _print_reference_guidance(self, cancer_type):

        print("\n========== REFERENCE GUIDANCE ==========")

        if self.h5ad_inputs:
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

    def run(self):

        normal = []
        tumor = []
        unspecified = []
        annotation_info = {}
        cancer_type = None

        tumor_adatas = []
        results = {}

        for gse_id in self.gse_ids:

            n, t, u, ann, ct = self._process_gse(gse_id)

            normal.extend(n)
            tumor.extend(t)
            unspecified.extend(u)

            annotation_info.update(ann)

            if ct and cancer_type is None:
                cancer_type = ct

            adata = self._build_h5ad(
                gse_id,
                t,
                save_single=(len(self.gse_ids) == 1 and len(self.h5ad_inputs) == 0)
            )

            if adata is not None:
                tumor_adatas.append(adata)

            results[gse_id] = (
                n,
                t,
                u,
                ann,
                None,
                ct
            )

        for file in self.h5ad_inputs:

            print("\n========== Reading h5ad file ==========")

            adata = sc.read_h5ad(file)

            adata.obs_names_make_unique()

            adata.layers["counts"] = adata.X.copy()
            adata.raw = adata

            tumor_adatas.append(adata)

            results[file] = (
                [],
                [],
                [],
                {},
                None,
                None
            )

        query_h5ad = None

        total_inputs = len(self.gse_ids) + len(self.h5ad_inputs)

        if total_inputs == 1:

            if len(self.gse_ids) == 1:

                query_h5ad = f"{self.gse_ids[0]}_tumor.h5ad"

                results[self.gse_ids[0]] = (
                    normal,
                    tumor,
                    unspecified,
                    annotation_info,
                    query_h5ad,
                    cancer_type
                )

            elif len(self.h5ad_inputs) == 1:

                adata = tumor_adatas[0]

                filename = "input_tumor.h5ad"

                adata.write(filename)

                print("\n========== h5ad created ==========")
                print(f"{filename} is created successfully")

                query_h5ad = filename

                key = self.h5ad_inputs[0]

                results[key] = (
                    [],
                    [],
                    [],
                    {},
                    query_h5ad,
                    None
                )

        elif total_inputs > 1 and len(tumor_adatas) > 0:

            combined = ad.concat(tumor_adatas, join="outer")

            combined.obs_names_make_unique()

            combined.layers["counts"] = combined.X.copy()
            combined.raw = combined

            combined.write("combined_tumor.h5ad")

            print("\n========== h5ad created ==========")
            print("combined_tumor.h5ad is created successfully")

            query_h5ad = "combined_tumor.h5ad"

            for key in results:

                n, t, u, ann, _, ct = results[key]

                results[key] = (
                    n,
                    t,
                    u,
                    ann,
                    query_h5ad,
                    ct
                )

        # ✅ NEW CALL
        self._print_reference_guidance(cancer_type)

        return normal, tumor, unspecified, annotation_info, query_h5ad, cancer_type, results

    def _process_gse(self, gse_id):

        gse_dir = os.path.join(self.base_dir, gse_id)

        os.makedirs(gse_dir, exist_ok=True)

        gse = GEOparse.get_GEO(
            geo=gse_id,
            destdir=gse_dir
        )

        gse.download_supplementary_files(gse_dir)

        normal = []
        tumor = []
        unspecified = []
        annotation_info = {}

        excluded_non_scrna = []
        excluded_non_human = []

        cancer_type = self._predict_cancer_type(gse)

        tumor_keywords = [
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
        print("Normal samples:", ", ".join(normal) if normal else "None")
        print("Tumor samples:", ", ".join(tumor) if tumor else "None")
        print("Unspecified samples:", ", ".join(unspecified) if unspecified else "None")
        print("Excluded (non-human):", ", ".join(excluded_non_human) if excluded_non_human else "None")
        print("Excluded (non-scRNA):", ", ".join(excluded_non_scrna) if excluded_non_scrna else "None")

        return normal, tumor, unspecified, annotation_info, cancer_type



    def _predict_cancer_type(self, gse):

        text = (
            gse.metadata.get("title", [""])[0] +
            " " +
            gse.metadata.get("summary", [""])[0]
        ).lower()

        cancer_map = {
            "ovary": ["ovarian"],
            "uterus": ["uterine", "endometrial"],
            "lung": ["lung"],
            "kidney": ["renal", "kidney"],
            "liver": ["liver", "hepatocellular"],
            "pancreas": ["pancreatic"],
            "prostate": ["prostate"],
            "bladder": ["bladder"],
            "stomach": ["gastric", "stomach"],
            "small_intestine": ["small intestine"],
            "large_intestine": ["colon", "colorectal"],
            "skin": ["melanoma", "skin"],
            "brain": ["glioma", "brain"],
            "blood": ["leukemia"],
            "lymph_node": ["lymphoma"],
            "bone_marrow": ["myeloma"],
            "heart": ["cardiac"],
            "thyroid": ["thyroid"],
            "esophagus": ["esophageal"],
            "trachea": ["trachea"],
            "tongue": ["tongue"],
            "salivary_gland": ["salivary"],
            "muscle": ["sarcoma"],
            "eye": ["ocular"],
            "ear": ["ear"],
            "fat": ["liposarcoma"],
            "vasculature": ["vascular"],
            "thymus": ["thymoma"],
            "testis": ["testicular"],
        }

        detected = []

        for tissue, keywords in cancer_map.items():
            if any(k in text for k in keywords):
                detected.append(f"{tissue}_cancer")

        if len(detected) == 0:
            return None

        return ", ".join(sorted(set(detected)))



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

            matrix_file = None
            features_file = None
            barcodes_file = None

            for f in files:
                if "matrix" in f and f.endswith(".mtx.gz"):
                    matrix_file = f
                elif "features" in f and f.endswith(".tsv.gz"):
                    features_file = f
                elif "barcodes" in f and f.endswith(".tsv.gz"):
                    barcodes_file = f

            adata = None

            if matrix_file and features_file and barcodes_file:

                try:
                    os.rename(os.path.join(gsm_dir, matrix_file),
                              os.path.join(gsm_dir, "matrix.mtx.gz"))
                except:
                    pass

                try:
                    os.rename(os.path.join(gsm_dir, features_file),
                              os.path.join(gsm_dir, "features.tsv.gz"))
                except:
                    pass

                try:
                    os.rename(os.path.join(gsm_dir, barcodes_file),
                              os.path.join(gsm_dir, "barcodes.tsv.gz"))
                except:
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

                    if any(f.endswith(ext) for ext in [".tsv", ".csv", ".txt", ".gz"]) \
                       and "matrix" in f.lower():

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

        # ✅ ONLY NEW CHANGE
        cancer_type = self._predict_cancer_type(
            GEOparse.get_GEO(geo=gse_id, destdir=self.base_dir)
        )
        if cancer_type is not None:
            combined.uns["cancer_type"] = cancer_type
            print(f"... stored cancer_type in h5ad: {cancer_type}")

        if save_single:

            filename = f"{gse_id}_tumor.h5ad"

            combined.write(filename)

            print("\n========== h5ad created ==========")
            print(f"{filename} is created successfully")

        return combined
