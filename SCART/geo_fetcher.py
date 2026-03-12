#!/usr/bin/env python

import os
import sys
import re
import logging
import gzip
import shutil
import tarfile
import pandas as pd
import scanpy as sc
from geofetch import GeoFetcher

# =========================================================
logging.getLogger().handlers.clear()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    datefmt="%d-%b-%Y %H:%M:%S"
)
# =========================================================


class GSMObject:
    """Minimal GSM-like object to mimic GEOparse structure"""

    def __init__(self, gsm_id, metadata):
        self.name = gsm_id
        self.metadata = metadata


class GSEObject:
    """Minimal GSE-like container"""

    def __init__(self):
        self.gsms = {}


class SampleAnnotator:

    def __init__(self, gse_id, data_dir="GSE_data"):

        if isinstance(gse_id, str):
            self.gse_ids = [gse_id]
        else:
            self.gse_ids = gse_id

        self.data_dir = data_dir
        self.gse = None
        self.current_gse_id = None

        self.normal_keywords = ["normal", "healthy"]
        self.tumor_keywords = [
            "tumor", "tumour", "cancer", "carcinoma",
            "adenocarcinoma", "malignant", "serous",
            "biopsy", "post-treatment", "post treatment"
        ]
        self.scrna_keywords = [
            "scrna-seq", "single-cell rna", "single cell rna",
            "single-cell transcriptome", "10x genomics",
            "chromium", "sc-rna"
        ]
        self.control_keyword = "control"

        self.excluded_metadata_keys = [
            "contact","organization","department","institute",
            "street","city","state","zip","country",
            "platform","series","submission","last_update",
            "email","phone"
        ]

        self.cancer_keywords = {
            "breast_cancer": ["breast","mammary"],
            "lung_cancer": ["lung","pulmonary"],
            "colorectal_cancer": ["colon","rectal","colorectal"],
            "liver_cancer": ["liver","hepatocellular"],
            "pancreatic_cancer": ["pancreas","pancreatic"],
            "ovarian_cancer": ["ovary","ovarian"],
            "prostate_cancer": ["prostate"],
            "brain_cancer": ["glioma","glioblastoma","brain"],
            "skin_cancer": ["melanoma","skin"],
            "blood_cancer": ["leukemia","lymphoma","myeloma"]
        }

        self.non_human_samples = []
        self.non_scrna_samples = []

    # =========================================================
    def run(self):

        all_results = {}

        for gid in self.gse_ids:

            self.current_gse_id = gid

            normal, tumor, unspecified, annotation_info = self.annotate_samples()

            cancer_type = self.predict_cancer_type()

            self._force_download_and_organize()

            query_h5ad = self.build_h5ad(tumor)

            all_results[gid] = (
                normal,
                tumor,
                unspecified,
                annotation_info,
                query_h5ad,
                cancer_type
            )

        if len(self.gse_ids) == 1:
            return all_results[self.gse_ids[0]]

        return all_results

    # =========================================================
    def predict_cancer_type(self):

        texts = []

        for gsm in self.gse.gsms.values():
            for v in gsm.metadata.values():
                texts.extend(v)

        combined = " ".join(texts).lower()

        for cancer, keywords in self.cancer_keywords.items():
            for k in keywords:
                if k in combined:
                    return cancer

        return "unknown"

    # =========================================================
    def _force_download_and_organize(self):

        os.makedirs(self.data_dir, exist_ok=True)

        gse_dir = os.path.join(self.data_dir, self.current_gse_id)

        fetcher = GeoFetcher(
            accession=self.current_gse_id,
            processed=True,
            just_metadata=False
        )

        fetcher.fetch_all(self.data_dir)

        metadata_file = os.path.join(
            self.data_dir,
            self.current_gse_id,
            "metadata",
            f"{self.current_gse_id}_samples.csv"
        )

        if not os.path.exists(metadata_file):
            raise RuntimeError("GEOfetch metadata not found")

        meta = pd.read_csv(metadata_file)

        self.gse = GSEObject()

        for _, row in meta.iterrows():

            gsm_id = row["gsm_name"]

            metadata = {}

            for col in meta.columns:
                val = str(row[col])
                metadata[col] = [val]

            self.gse.gsms[gsm_id] = GSMObject(gsm_id, metadata)

        logging.info(
            f"Loaded GEO: {self.current_gse_id}, "
            f"Total samples: {len(self.gse.gsms)}"
        )

        for root, dirs, files in os.walk(self.data_dir):
            for f in files:
                path = os.path.join(root, f)
                try:
                    if f.endswith(".tar") or f.endswith(".tar.gz"):
                        extract_dir = path + "_extracted"
                        if not os.path.exists(extract_dir):
                            os.makedirs(extract_dir, exist_ok=True)
                            with tarfile.open(path) as tar:
                                tar.extractall(extract_dir)
                    elif f.endswith(".gz") and not f.endswith(".tar.gz"):
                        out_file = path.replace(".gz", "")
                        if not os.path.exists(out_file):
                            with gzip.open(path, "rb") as f_in:
                                with open(out_file, "wb") as f_out:
                                    shutil.copyfileobj(f_in, f_out)
                except Exception as e:
                    logging.warning(f"Extraction failed for {path}: {e}")

        gse_dir = os.path.join(self.data_dir, self.current_gse_id)
        os.makedirs(gse_dir, exist_ok=True)

        for gsm_id in self.gse.gsms.keys():

            gsm_dir = os.path.join(gse_dir, gsm_id)
            os.makedirs(gsm_dir, exist_ok=True)

            for root, dirs, files in os.walk(self.data_dir):
                for f in files:
                    if gsm_id in f:
                        src = os.path.join(root, f)
                        dst = os.path.join(gsm_dir, f)

                        if not os.path.exists(dst):
                            shutil.move(src, dst)

        for gsm_id in os.listdir(gse_dir):

            gsm_dir = os.path.join(gse_dir, gsm_id)

            if not os.path.isdir(gsm_dir):
                continue

            for f in os.listdir(gsm_dir):

                src = os.path.join(gsm_dir, f)

                if "matrix" in f and f.endswith(".mtx.gz"):
                    shutil.move(src, os.path.join(gsm_dir, "matrix.mtx.gz"))

                elif "features" in f and f.endswith(".tsv.gz"):
                    shutil.move(src, os.path.join(gsm_dir, "features.tsv.gz"))

                elif "barcodes" in f and f.endswith(".tsv.gz"):
                    shutil.move(src, os.path.join(gsm_dir, "barcodes.tsv.gz"))

    # =========================================================
    def annotate_samples(self):

        self._force_download_and_organize()

        self.non_human_samples = []
        self.non_scrna_samples = []

        normal = []
        tumor = []
        unspecified = []
        annotation_info = {}

        for gsm_id, gsm in self.gse.gsms.items():

            label = "unspecified"

            text_blob = " ".join(
                [" ".join(v) for v in gsm.metadata.values()]
            ).lower()

            if any(k in text_blob for k in self.normal_keywords):
                label = "normal"

            if any(k in text_blob for k in self.tumor_keywords):
                label = "tumor"

            annotation_info[gsm_id] = {"label": label}

            if label == "normal":
                normal.append(gsm_id)
            elif label == "tumor":
                tumor.append(gsm_id)
            else:
                unspecified.append(gsm_id)

        return normal, tumor, unspecified, annotation_info

    # =========================================================
    def build_h5ad(self, tumor_samples):

        adata_list = []

        for gsm in tumor_samples:

            sample_dir = os.path.join(
                self.data_dir,
                self.current_gse_id,
                gsm
            )

            if not os.path.exists(sample_dir):
                print(f"Skipping {gsm} → no directory")
                continue

            mtx_files = [f for f in os.listdir(sample_dir) if ".mtx" in f]
            if not mtx_files:
                print(f"Skipping {gsm} → no matrix files found")
                continue

            try:
                ad = sc.read_10x_mtx(sample_dir, var_names="gene_symbols", make_unique=True)
            except Exception as e:
                logging.warning(f"Failed to read 10X matrix for {gsm}: {e}")
                continue

            ad.obs["gsm_id"] = gsm
            ad.obs["gse_id"] = self.current_gse_id

            adata_list.append(ad)

        if not adata_list:
            logging.warning("No tumor matrices found across GSEs")
            return None

        combined = adata_list[0].concatenate(adata_list[1:], join="outer")

        output_path = os.path.join(
            self.data_dir,
            self.current_gse_id,
            f"{self.current_gse_id}_tumor.h5ad"
        )

        combined.write(output_path)

        return output_path