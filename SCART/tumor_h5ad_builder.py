import os
import scanpy as sc
import anndata as ad


class TumorH5ADBuilder:

    def __init__(self, data_dir="GSE_data"):
        self.data_dir = data_dir

    def build(self, results):

        all_adatas = []

        for gse_id, res in results.items():

            normal, tumor, unspecified, annotation, query_h5ad, cancer = res

            print(f"\n========== SAMPLE SUMMARY: {gse_id} ==========")
            print(f"Cancer type: {cancer}")

            print(f"Normal samples: {', '.join(normal) if normal else 'None'}")
            print(f"Tumor samples: {', '.join(tumor) if tumor else 'None'}")
            print(f"Unspecified samples: {', '.join(unspecified) if unspecified else 'None'}")

            gse_dir = os.path.join(self.data_dir, gse_id)

            print("\n========== Reading Tumor Samples ==========")

            for gsm in tumor:

                gsm_dir = os.path.join(gse_dir, gsm)

                if not os.path.exists(gsm_dir):
                    print(f"Skipping {gsm} → folder not found")
                    continue

                try:

                    adata = sc.read_10x_mtx(
                        gsm_dir,
                        var_names="gene_symbols",
                        make_unique=True
                    )

                except Exception:

                    print(f"Skipping {gsm} → invalid matrix")
                    continue

                adata.obs_names = [
                    f"{gsm}_{x}" for x in adata.obs_names
                ]

                adata.obs["gsm_id"] = gsm
                adata.obs["gse_id"] = gse_id

                all_adatas.append(adata)

        if not all_adatas:
            print("No tumor samples found")
            return None

        combined = ad.concat(all_adatas, join="outer")

        combined.layers["counts"] = combined.X.copy()
        combined.raw = combined

        out = os.path.join(self.data_dir,"combined_tumor.h5ad")

        combined.write(out)

        print("\n========== h5ad created ==========")
        print("combined_tumor.h5ad is created successfully")

        return out