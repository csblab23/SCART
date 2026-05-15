"""
SCART — CAR-T Target Discovery
Streamlit web application
Connects directly to all SCART Python modules.
"""

import os
import sys
import streamlit as st

# ── Make SCART modules importable ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="SCART — CAR-T Target Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS injected into Streamlit ────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] {
    background: #050810;
    color: #e8f0fe;
}
[data-testid="stSidebar"] {
    background: #0a0e1a;
    border-right: 1px solid rgba(99,168,255,0.12);
}
[data-testid="stSidebar"] * { color: #8fa3c8 !important; }
[data-testid="stSidebar"] .sidebar-logo { color: #4f9cf9 !important; }

/* ── Headings ── */
h1, h2, h3 { color: #e8f0fe !important; font-weight: 600 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #0a0e1a;
    border: 1px solid rgba(99,168,255,0.12);
    border-radius: 12px;
    padding: 16px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #4f9cf9;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea,
.stNumberInput input, .stSelectbox select {
    background: #0f1422 !important;
    border: 1px solid rgba(99,168,255,0.2) !important;
    color: #e8f0fe !important;
    border-radius: 8px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0a0e1a;
    border-bottom: 1px solid rgba(99,168,255,0.12);
}
.stTabs [data-baseweb="tab"] { color: #8fa3c8; }
.stTabs [aria-selected="true"] { color: #4f9cf9 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(99,168,255,0.12);
    border-radius: 12px;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0a0e1a;
    border: 1px solid rgba(99,168,255,0.12);
    border-radius: 10px;
}

/* ── Log / code boxes ── */
.log-box {
    background: #000;
    border: 1px solid rgba(99,168,255,0.15);
    border-radius: 10px;
    padding: 14px;
    font-family: monospace;
    font-size: 12px;
    color: #22c55e;
    line-height: 1.8;
    max-height: 220px;
    overflow-y: auto;
}

/* ── Info / success / warning banners ── */
.stAlert { border-radius: 10px !important; }

/* ── Divider ── */
hr { border-color: rgba(99,168,255,0.12) !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE  — carries results between modules
# ════════════════════════════════════════════════════════════════════════════
defaults = {
    "tumor_h5ad":       None,   # path written by Module 1
    "popv_h5ad":        None,   # path written by Module 2
    "final_h5ad":       None,   # path written by Module 3
    "single_gene_df":   None,   # DataFrame from Module 4a
    "two_gene_df":      None,   # DataFrame from Module 4b
    "chat_history":     [],     # Gemini chat messages
    "gemini_key":       "",     # API key (in-session only)
    "pipeline_status":  {       # track which steps are done
        "geo":   False,
        "annot": False,
        "malig": False,
        "score": False,
    },
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧬 SCART")
    st.caption("CAR-T Target Discovery v2.0")
    st.divider()

    # Pipeline status indicators
    def _badge(done): return "✅" if done else "⬜"
    st.markdown(f"""
    **Pipeline status**
    {_badge(st.session_state.pipeline_status['geo'])}  Module 1 — GEO Fetcher
    {_badge(st.session_state.pipeline_status['annot'])} Module 2 — Cell Annotation
    {_badge(st.session_state.pipeline_status['malig'])} Module 3 — Malignant ID
    {_badge(False)} Module 4 — Surfaceome DE *(in module 3)*
    {_badge(st.session_state.pipeline_status['score'])} Module 5 — Gene Scoring
    """)
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠  Dashboard",
            "1️⃣  GEO Fetcher",
            "2️⃣  Cell Annotation",
            "3️⃣  Malignant ID + Surfaceome DE",
            "5️⃣  Gene Scoring",
            "🤖  Gemini Advisor",
            "📥  Results & Export",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("PopV · scMalignantFinder · SCEVAN · DEAP GA")


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════
def module_header(title, subtitle):
    st.title(title)
    st.markdown(f"<p style='color:#8fa3c8;font-size:15px;margin-bottom:24px'>{subtitle}</p>",
                unsafe_allow_html=True)

def info_card(label, value, color="#4f9cf9"):
    st.markdown(f"""
    <div style='background:#0a0e1a;border:1px solid rgba(99,168,255,0.15);
    border-radius:12px;padding:16px 20px;margin-bottom:8px'>
    <div style='font-size:11px;color:#4d6080;text-transform:uppercase;
    letter-spacing:.8px;margin-bottom:4px'>{label}</div>
    <div style='font-size:22px;font-weight:700;color:{color}'>{value}</div>
    </div>""", unsafe_allow_html=True)

def success_banner(msg):
    st.markdown(f"""
    <div style='background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.25);
    border-radius:10px;padding:12px 16px;color:#22c55e;font-size:14px;margin:8px 0'>
    ✅ {msg}</div>""", unsafe_allow_html=True)

def warn_banner(msg):
    st.markdown(f"""
    <div style='background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);
    border-radius:10px;padding:12px 16px;color:#f59e0b;font-size:14px;margin:8px 0'>
    ⚠️ {msg}</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠  Dashboard":
    module_header(
        "CAR-T Target Discovery",
        "End-to-end pipeline from scRNA-seq → tumour-specific surface protein targets. "
        "Run modules sequentially or resume from any step with pre-processed data."
    )

    # Metric row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Datasets loaded",     "—" if not st.session_state.tumor_h5ad  else "✓")
    c2.metric("Annotation done",     "—" if not st.session_state.popv_h5ad   else "✓")
    c3.metric("Malignant cells",     "—" if not st.session_state.final_h5ad  else "✓")
    c4.metric("Targets scored",      "—" if st.session_state.single_gene_df is None else
              f"{len(st.session_state.single_gene_df)} genes")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Pipeline status")
        steps = [
            ("1 — GEO Fetcher",              st.session_state.pipeline_status["geo"],
             "Download + QC scRNA-seq datasets"),
            ("2 — Cell Annotation",          st.session_state.pipeline_status["annot"],
             "PopV consensus, 18 cell types"),
            ("3 — Malignant ID + Surf. DE",  st.session_state.pipeline_status["malig"],
             "scMalignantFinder + SCEVAN + DEG"),
            ("5 — Gene Scoring",             st.session_state.pipeline_status["score"],
             "Efficacy + safety, AND/OR/NOT gates"),
        ]
        for name, done, detail in steps:
            icon = "✅" if done else "🔵"
            st.markdown(
                f"{icon} **{name}**  \n"
                f"<span style='color:#4d6080;font-size:12px'>{detail}</span>",
                unsafe_allow_html=True,
            )

    with right:
        st.subheader("How to use SCART")
        st.markdown("""
1. **GEO Fetcher** — paste GEO accession IDs or upload h5ad
2. **Cell Annotation** — run PopV (or skip with your own labels)
3. **Malignant ID** — scMalignantFinder + SCEVAN + surfaceome DEG
4. **Gene Scoring** — single gene and two-gene combinations
5. **Export** — download CSV, h5ad, and YAML config
        """)
        st.markdown("")
        if st.button("▶️  Start analysis — go to Module 1"):
            st.info("Select **1️⃣  GEO Fetcher** from the sidebar.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: MODULE 1 — GEO FETCHER
# ════════════════════════════════════════════════════════════════════════════
elif page == "1️⃣  GEO Fetcher":
    module_header(
        "Module 1 — GEO Fetcher",
        "Download scRNA-seq datasets from NCBI GEO by accession ID, or upload your own "
        "pre-processed h5ad file. QC parameters are stored and passed to downstream modules."
    )

    tab_geo, tab_upload = st.tabs(["📡 Fetch from GEO", "📂 Upload h5ad"])

    # ── TAB 1: GEO download ──────────────────────────────────────────────
    with tab_geo:
        col1, col2 = st.columns(2)

        with col1:
            geo_ids_raw = st.text_area(
                "GEO Accession IDs (one per line)",
                value="GSE162499\nGSE144735",
                height=110,
                help="Supports GSE and GSM formats"
            )

            try:
                from SCART.geo_fetcher import VALID_CANCER_TYPES
                cancer_opts = VALID_CANCER_TYPES
            except Exception:
                cancer_opts = ["lung_cancer", "blood_cancer", "ovary_cancer",
                               "pancreas_cancer", "breast_cancer", "kidney_cancer"]

            cancer_type = st.selectbox("Cancer type", cancer_opts)

        with col2:
            min_genes = st.number_input("Min genes / cell", value=200, step=50,
                                        help="Cells with fewer genes are removed")
            max_mt    = st.number_input("Max % mitochondrial reads", value=20, step=5,
                                        help="Cells with higher MT% are removed")
            doublet   = st.checkbox("Doublet removal (Scrublet)", value=True)
            ambient   = st.checkbox("Ambient RNA correction (SoupX)", value=False)

        st.markdown("")
        run_geo = st.button("🚀 Fetch Datasets", type="primary", use_container_width=True)

        if run_geo:
            ids = [x.strip() for x in geo_ids_raw.strip().split("\n") if x.strip()]
            if not ids:
                st.error("Enter at least one GEO accession ID.")
            else:
                with st.status("Fetching datasets from NCBI GEO…", expanded=True) as status:
                    try:
                        st.write(f"📡 Connecting to GEO for: {', '.join(ids)}")
                        from SCART.geo_fetcher import SampleAnnotator

                        annotator = SampleAnnotator(
                            *ids,
                            cancer_type=cancer_type,
                            min_genes=int(min_genes) if min_genes else None,
                            max_mt=float(max_mt)     if max_mt    else None,
                        )
                        results = annotator.run()
                        normal, tumor, unspecified, ann_info, query_h5ad, ct, _ = results

                        st.write(f"✅ Tumor samples:      {len(tumor)}")
                        st.write(f"✅ Normal samples:     {len(normal)}")
                        st.write(f"✅ Unspecified:        {len(unspecified)}")
                        st.write(f"✅ Output file:        {query_h5ad}")

                        st.session_state.tumor_h5ad = query_h5ad
                        st.session_state.pipeline_status["geo"] = True
                        status.update(label="Done!", state="complete")

                        success_banner(f"h5ad saved → {query_h5ad}")

                        # Show sample table
                        import pandas as pd
                        rows = [(k, v) for k, v in ann_info.items()]
                        if rows:
                            df_ann = pd.DataFrame(rows, columns=["GSM ID", "Label"])
                            st.dataframe(df_ann, use_container_width=True)

                    except Exception as e:
                        status.update(label="Error", state="error")
                        st.error(f"Error: {e}")
                        st.exception(e)

    # ── TAB 2: Upload h5ad ───────────────────────────────────────────────
    with tab_upload:
        st.markdown("Upload a pre-processed h5ad file with **raw counts** in `.X` or `layers['counts']`.")

        uploaded = st.file_uploader("Choose h5ad file", type=["h5ad"])
        manual_col = st.text_input(
            "Manual annotation column (optional)",
            placeholder="e.g. cell_type  — leave blank to run PopV in Module 2",
            help="If your h5ad already has cell type labels, enter the obs column name here. PopV will be skipped."
        )

        if uploaded:
            save_path = uploaded.name
            if st.button("✅ Use this file", type="primary"):
                with st.spinner("Saving uploaded file…"):
                    with open(save_path, "wb") as f:
                        f.write(uploaded.read())

                    # If manual annotation: run SampleAnnotator with h5ad path
                    try:
                        from SCART.geo_fetcher import SampleAnnotator
                        kwargs = dict(cancer_type=cancer_type)
                        if manual_col.strip():
                            kwargs["manual_annotation_col"] = manual_col.strip()

                        annotator = SampleAnnotator(save_path, **kwargs)
                        annotator.run()
                        out = "input_tumor.h5ad"

                    except Exception:
                        # Fallback: just save path directly
                        out = save_path

                    st.session_state.tumor_h5ad = out
                    st.session_state.pipeline_status["geo"] = True
                    success_banner(f"File ready → {out}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: MODULE 2 — CELL ANNOTATION
# ════════════════════════════════════════════════════════════════════════════
elif page == "2️⃣  Cell Annotation":
    module_header(
        "Module 2 — Cell Type Annotation",
        "PopV multi-method consensus: KNN, SVM, scANVI, logistic regression. "
        "Tabula Sapiens reference is auto-downloaded based on your cancer type. "
        "Skip this step if you already have annotations."
    )

    if not st.session_state.pipeline_status["geo"]:
        warn_banner("Complete Module 1 (GEO Fetcher) first — or set tumor_h5ad path below.")

    input_path = st.text_input(
        "Input h5ad path",
        value=st.session_state.tumor_h5ad or "",
        help="Output of Module 1, e.g. GSE162499_tumor.h5ad or combined_tumor.h5ad"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("PopV settings")
        user_ref = st.text_input(
            "Custom reference h5ad (optional)",
            placeholder="Leave blank → auto-download Tabula Sapiens",
            help="Download from: https://doi.org/10.6084/m9.figshare.27921984"
        )
        n_samples = st.slider("Cells sampled per label", 100, 500, 300,
                              help="Higher = better accuracy, slower runtime")
        drop_ref  = st.checkbox("Remove Tabula Sapiens metadata columns from output", True)
        n_jobs    = st.number_input("CPU cores", min_value=1, value=1, step=1)

    with col2:
        st.subheader("Skip / reuse options")
        user_pred = st.text_input(
            "Already have PopV result? Paste path here (skips re-running)",
            placeholder="e.g. popv_results/final_popv_annotated.h5ad"
        )
        st.markdown("")
        st.info(
            "**Skip PopV entirely** if you provided a manual annotation column "
            "in Module 1 — `adata.uns['skip_popv']` will be `True` automatically."
        )

    st.markdown("")
    run_annot = st.button("▶️  Run Annotation", type="primary", use_container_width=True)

    if run_annot:
        with st.spinner("Running PopV annotation… (this can take 20–60 min on first run)"):
            try:
                from SCART.popv_annotation import auto_run_popv

                result = auto_run_popv(
                    nsamples             = int(n_samples),
                    output_dir           = "popv_results",
                    user_reference       = user_ref  or None,
                    user_popv_prediction = user_pred or None,
                    drop_reference_columns = drop_ref,
                    n_jobs               = int(n_jobs),
                )

                out_path = "popv_results/final_popv_annotated.h5ad"
                st.session_state.popv_h5ad = out_path
                st.session_state.pipeline_status["annot"] = True
                success_banner(f"Annotation complete → {out_path}  ({result.n_obs:,} cells)")

                # Cell type bar chart
                if "popv_majority_vote_prediction" in result.obs.columns:
                    import pandas as pd
                    counts = (result.obs["popv_majority_vote_prediction"]
                              .value_counts().reset_index())
                    counts.columns = ["Cell type", "Count"]

                    st.subheader("Cell type composition")
                    st.bar_chart(counts.set_index("Cell type"))

                    col_a, col_b = st.columns(2)
                    col_a.metric("Total cells annotated", f"{result.n_obs:,}")
                    col_b.metric("Unique cell types",
                                 result.obs["popv_majority_vote_prediction"].nunique())

            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: MODULE 3 — MALIGNANT ID + SURFACEOME DE
# ════════════════════════════════════════════════════════════════════════════
elif page == "3️⃣  Malignant ID + Surfaceome DE":
    module_header(
        "Module 3 — Malignant ID + Surfaceome DE",
        "scMalignantFinder (CNV inference) cross-validated with SCEVAN. "
        "Identifies malignant epithelial cells, then runs Wilcoxon DEG against "
        "non-epithelial cells, filtered to the CSPA surfaceome."
    )

    if not st.session_state.pipeline_status["annot"]:
        warn_banner("Complete Module 2 (Cell Annotation) first — or set popv_h5ad path below.")

    popv_path = st.text_input(
        "PopV-annotated h5ad path",
        value=st.session_state.popv_h5ad or "popv_results/final_popv_annotated.h5ad",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Malignancy detection")
        strategy = st.selectbox(
            "Strategy",
            ["intersection", "scMalignant", "scevan"],
            help="intersection = scMalignantFinder AND SCEVAN must both agree"
        )
        reference_h5ad = st.text_input(
            "Reference h5ad for SCEVAN",
            placeholder="e.g. popv_reference/Lung_TSP1_30_...h5ad",
            help="Tabula Sapiens tissue h5ad used as normal epithelial reference for CNV inference"
        )
        ref_max_cells = st.number_input("Max SCEVAN reference cells", value=500, step=100)
        scevan_cores  = st.number_input("SCEVAN CPU cores (par_cores)", value=1, step=1)

    with col2:
        st.subheader("DEG thresholds")
        log2fc = st.number_input("Log₂ fold-change threshold", value=1.0, step=0.1,
                                 help="Minimum log2FC for a gene to be called DE")
        padj   = st.number_input("Adjusted p-value threshold", value=0.05, step=0.01,
                                 help="Benjamini-Hochberg corrected p-value cutoff")

        st.subheader("Paths (auto-detected if blank)")
        scm_model = st.text_input(
            "scMalignantFinder model dir",
            placeholder="auto-detected from SCART package"
        )
        surf_path = st.text_input(
            "Surfaceome CSV path",
            placeholder="auto-detected: SCART/GESP/GESP_surfaceome_gene.csv"
        )

    st.markdown("")
    run_malig = st.button("▶️  Run Malignant ID + DEG", type="primary", use_container_width=True)

    if run_malig:
        with st.status("Running malignancy detection…", expanded=True) as status:
            try:
                from SCART.preprocessing import run_preprocessing_pipeline

                st.write("📍 Loading PopV-annotated h5ad…")
                kwargs = dict(
                    popv_path          = popv_path            or None,
                    reference_h5ad     = reference_h5ad       or None,
                    log2fc_threshold   = float(log2fc),
                    pval_adj_threshold = float(padj),
                    malignant_strategy = strategy,
                    scevan_ref_max_cells = int(ref_max_cells),
                    scevan_par_cores   = int(scevan_cores),
                )
                if scm_model.strip():
                    kwargs["scmalignant_model_dir"] = scm_model.strip()
                if surf_path.strip():
                    kwargs["surfaceome_path"] = surf_path.strip()

                st.write("🔬 Running scMalignantFinder…")
                result = run_preprocessing_pipeline(**kwargs)

                out_path = "preprocessing_results/final_tumor.h5ad"
                st.session_state.final_h5ad = out_path
                st.session_state.pipeline_status["malig"] = True
                status.update(label="Done!", state="complete")

                # Summary metrics
                deg_params = result.uns.get("deg_params", {})
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Malignant cells",    f"{result.n_obs:,}")
                c2.metric("Surfaceome DEGs",     deg_params.get("n_filtered_deg", "—"))
                c3.metric("Genes tested",        deg_params.get("n_surfaceome_genes", "—"))
                c4.metric("Normal cells (rest)", deg_params.get("n_rest", "—"))

                success_banner(f"Preprocessing complete → {out_path}")

                # Show top DEGs
                if "filtered_deg" in result.uns:
                    import pandas as pd
                    deg_df = result.uns["filtered_deg"]
                    if isinstance(deg_df, pd.DataFrame) and len(deg_df) > 0:
                        st.subheader("Top differentially expressed surface genes")
                        st.dataframe(
                            deg_df[["names","logfoldchanges","pvals_adj","pct_nz_group","pct_nz_reference"]]
                            .head(30).rename(columns={
                                "names":          "Gene",
                                "logfoldchanges": "log2FC",
                                "pvals_adj":      "adj. p-value",
                                "pct_nz_group":   "% tumour",
                                "pct_nz_reference": "% healthy",
                            }),
                            use_container_width=True,
                            column_config={
                                "log2FC":       st.column_config.NumberColumn(format="%.3f"),
                                "adj. p-value": st.column_config.NumberColumn(format="%.2e"),
                                "% tumour":     st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.1%"),
                                "% healthy":    st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.1%"),
                            }
                        )

            except Exception as e:
                status.update(label="Error", state="error")
                st.error(f"Error: {e}")
                st.exception(e)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: MODULE 5 — GENE SCORING
# ════════════════════════════════════════════════════════════════════════════
elif page == "5️⃣  Gene Scoring":
    module_header(
        "Module 5 — Gene Combination Scoring",
        "Score every surface gene individually and every two-gene AND/OR/NOT "
        "logic-gate combination for efficacy (tumour coverage) and safety (healthy tissue sparing). "
        "Two-gene pairs are searched using a Genetic Algorithm."
    )

    if not st.session_state.pipeline_status["malig"]:
        warn_banner("Complete Module 3 first — or set the tumor h5ad path below.")

    tumor_path = st.text_input(
        "Final tumor h5ad path",
        value=st.session_state.final_h5ad or "preprocessing_results/final_tumor.h5ad",
    )
    hpa_path = st.text_input(
        "HPA healthy reference (optional)",
        placeholder="Leave blank to auto-download from proteinatlas.org",
        help="Path to HPA h5ad or TSV file. Auto-downloaded if blank."
    )

    tab_single, tab_two = st.tabs(["🧬 Single gene", "🔗 Two-gene combinations (GA)"])

    # ── Single gene ──────────────────────────────────────────────────────
    with tab_single:
        safety_thresh = st.slider("Safety threshold", 0.5, 1.0, 0.9, 0.01,
                                  help="Minimum fraction of healthy cells NOT expressing the gene")

        run_single = st.button("▶️  Score all surface genes", type="primary")

        if run_single:
            with st.spinner("Scoring all surfaceome genes against HPA healthy reference…"):
                try:
                    from SCART.gene_combination_predictor.one_gene_combination import run

                    df = run(
                        safety_threshold = safety_thresh,
                        hpa_path         = hpa_path  or None,
                        tumor_path       = tumor_path or None,
                    )
                    st.session_state.single_gene_df = df
                    st.session_state.pipeline_status["score"] = True

                    top = (df[df["Safety"] >= safety_thresh]
                           .sort_values("Efficacy", ascending=False))

                    success_banner(f"Scored {len(df):,} genes — "
                                   f"{len(top)} pass safety ≥ {safety_thresh:.0%}")

                    st.subheader(f"Top candidates (safety ≥ {safety_thresh:.0%})")
                    st.dataframe(
                        top.head(50),
                        use_container_width=True,
                        column_config={
                            "Efficacy": st.column_config.ProgressColumn(
                                min_value=0, max_value=1, format="%.3f"),
                            "Safety":   st.column_config.ProgressColumn(
                                min_value=0, max_value=1, format="%.3f"),
                        }
                    )

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.exception(e)

        # Show cached result
        elif st.session_state.single_gene_df is not None:
            df = st.session_state.single_gene_df
            st.info(f"Showing cached result — {len(df):,} genes scored.")
            st.dataframe(df.sort_values("Efficacy", ascending=False).head(30),
                         use_container_width=True)

    # ── Two-gene GA ──────────────────────────────────────────────────────
    with tab_two:
        col1, col2 = st.columns(2)
        with col1:
            safety_t2 = st.slider("Safety threshold (two-gene)", 0.5, 1.0, 0.9, 0.01)
            pop_size  = st.number_input("GA population size", value=500, step=100,
                                        help="Larger = better results, slower")
            n_gen     = st.number_input("Max generations", value=50, step=10)
        with col2:
            n_runs    = st.number_input("Independent GA runs", value=5, step=1,
                                        help="More runs = more diverse results")
            n_cpus    = st.number_input("CPU cores", value=1, step=1)
            patience  = st.number_input("Early-stop patience (generations)", value=20, step=5)

        st.info("💡 Two-gene scoring explores all AND / OR / NOT gate combinations. "
                "Typical runtime: 30–90 min for pop_size=1000.")

        run_two = st.button("▶️  Run GA two-gene search", type="primary")

        if run_two:
            with st.spinner("Running genetic algorithm… check terminal for progress"):
                try:
                    from SCART.gene_combination_predictor.two_gene_combination import run

                    df_hof, df_all = run(
                        hpa_path         = hpa_path   or None,
                        tumor_path       = tumor_path or None,
                        safety_threshold = safety_t2,
                        pop_size         = int(pop_size),
                        Gmax             = int(n_gen),
                        n_runs           = int(n_runs),
                        n_cpus           = int(n_cpus),
                        patience         = int(patience),
                    )
                    st.session_state.two_gene_df = df_hof
                    st.session_state.pipeline_status["score"] = True

                    success_banner(f"GA complete — {len(df_hof)} unique combinations in Hall of Fame")

                    st.subheader("Hall of Fame — top two-gene combinations")
                    st.dataframe(
                        df_hof.head(30),
                        use_container_width=True,
                        column_config={
                            "Efficacy": st.column_config.ProgressColumn(
                                min_value=0, max_value=1, format="%.3f"),
                            "Safety":   st.column_config.ProgressColumn(
                                min_value=0, max_value=1, format="%.3f"),
                        }
                    )

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.exception(e)

        elif st.session_state.two_gene_df is not None:
            st.info("Showing cached result.")
            st.dataframe(st.session_state.two_gene_df.head(30),
                         use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: GEMINI ADVISOR
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Gemini Advisor":
    module_header(
        "Gemini AI Advisor",
        "Ask anything about your SCART analysis — target rationale, literature context, "
        "experimental design, or biological mechanisms. Powered by Gemini API (free tier)."
    )

    # API key input
    with st.expander("🔑  Gemini API Key", expanded=not st.session_state.gemini_key):
        key_input = st.text_input(
            "API key",
            type="password",
            value=st.session_state.gemini_key,
            placeholder="AIzaSy…",
            help="Free key at https://aistudio.google.com/apikey — stored in-session only"
        )
        if st.button("Save key"):
            st.session_state.gemini_key = key_input
            st.success("Key saved for this session.")

    st.divider()

    # Quick question chips
    st.markdown("**Quick questions:**")
    quick_cols = st.columns(3)
    quick_qs = [
        "Why is MSLN a good CAR-T target?",
        "What are the risks of HER2-targeted CAR-T?",
        "How does an AND gate improve safety?",
        "What MSLN clinical trials exist?",
        "Design a validation experiment for MSLN+CD70",
        "FOLR1 vs FOLR2 in tumour vs normal tissue",
    ]
    for i, q in enumerate(quick_qs):
        with quick_cols[i % 3]:
            if st.button(q, key=f"quick_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": q})

    st.divider()

    # Chat history display
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    prompt = st.chat_input("Ask anything about your SCART results or CAR-T biology…")

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if not st.session_state.gemini_key:
            st.warning("Enter your Gemini API key above to enable AI responses.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=st.session_state.gemini_key)
                        model = genai.GenerativeModel("gemini-2.0-flash")

                        # Build context from current session
                        ctx_parts = [
                            "You are an expert computational biologist specialising in "
                            "CAR-T cell therapy and single-cell RNA-seq analysis. "
                            "The user is running SCART, a pipeline that:\n"
                            "1. Downloads scRNA-seq from GEO and performs QC\n"
                            "2. Annotates cell types with PopV (KNN, SVM, scANVI consensus)\n"
                            "3. Identifies malignant cells with scMalignantFinder + SCEVAN\n"
                            "4. Computes surfaceome DEG (malignant vs non-malignant)\n"
                            "5. Scores single genes and AND/OR/NOT two-gene combinations\n"
                        ]
                        if st.session_state.single_gene_df is not None:
                            top5 = (st.session_state.single_gene_df
                                    .sort_values("Efficacy", ascending=False).head(5))
                            ctx_parts.append(
                                f"Current top single-gene results:\n{top5.to_string()}\n"
                            )
                        if st.session_state.two_gene_df is not None:
                            top3 = st.session_state.two_gene_df.head(3)
                            ctx_parts.append(
                                f"Current top two-gene results:\n{top3.to_string()}\n"
                            )
                        ctx_parts.append(f"\nUser question: {prompt}")

                        response = model.generate_content("\n".join(ctx_parts))
                        reply = response.text

                        st.write(reply)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": reply}
                        )

                    except Exception as e:
                        st.error(f"Gemini error: {e}")

    if st.button("🗑️  Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS & EXPORT
# ════════════════════════════════════════════════════════════════════════════
elif page == "📥  Results & Export":
    module_header(
        "Results & Export",
        "Download your complete SCART analysis — ranked target lists, "
        "annotated h5ad files, and pipeline config for reproducibility."
    )

    import pandas as pd

    # ── Summary ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    single_df = st.session_state.single_gene_df
    two_df    = st.session_state.two_gene_df

    c1.metric("Single genes scored",
              f"{len(single_df):,}" if single_df is not None else "—")
    c2.metric("Two-gene combos (HoF)",
              f"{len(two_df):,}"    if two_df    is not None else "—")
    c3.metric("Pipeline steps done",
              sum(st.session_state.pipeline_status.values()))

    st.divider()

    # ── File downloads ────────────────────────────────────────────────────
    st.subheader("📁 Download files")

    download_items = [
        {
            "label":    "Single-gene results",
            "path":     "single_gene_results.csv",
            "df":       single_df,
            "filename": "SCART_single_gene_results.csv",
            "mime":     "text/csv",
        },
        {
            "label":    "Two-gene Hall of Fame",
            "path":     "two_gene_hof.csv",
            "df":       two_df,
            "filename": "SCART_two_gene_hof.csv",
            "mime":     "text/csv",
        },
    ]

    for item in download_items:
        col_a, col_b = st.columns([4, 1])
        col_a.markdown(f"**{item['label']}** — `{item['filename']}`")
        # Prefer in-memory DataFrame, then fall back to file on disk
        if item["df"] is not None:
            csv_bytes = item["df"].to_csv(index=False).encode()
            col_b.download_button(
                "⬇️ Download",
                data=csv_bytes,
                file_name=item["filename"],
                mime=item["mime"],
                key=item["filename"],
            )
        elif os.path.exists(item["path"]):
            with open(item["path"], "rb") as f:
                col_b.download_button(
                    "⬇️ Download",
                    data=f.read(),
                    file_name=item["filename"],
                    mime=item["mime"],
                    key=item["filename"],
                )
        else:
            col_b.caption("Not generated yet")

    # h5ad files
    for h5ad_path, label in [
        (st.session_state.final_h5ad,  "Final tumor h5ad (malignant + DEG)"),
        (st.session_state.popv_h5ad,   "PopV-annotated h5ad"),
        (st.session_state.tumor_h5ad,  "Raw tumor h5ad (Module 1 output)"),
    ]:
        if h5ad_path and os.path.exists(h5ad_path):
            col_a, col_b = st.columns([4, 1])
            col_a.markdown(f"**{label}** — `{os.path.basename(h5ad_path)}`")
            with open(h5ad_path, "rb") as f:
                col_b.download_button(
                    "⬇️ Download",
                    data=f.read(),
                    file_name=os.path.basename(h5ad_path),
                    mime="application/octet-stream",
                    key=h5ad_path,
                )

    st.divider()

    # ── Pipeline YAML config ──────────────────────────────────────────────
    st.subheader("📋 Pipeline config (YAML)")
    st.markdown("Copy this to reproduce your exact analysis run.")
    yaml_config = """geo_accessions: []          # add your GSE IDs
cancer_type: lung_cancer
qc:
  min_genes: 200
  max_mt_pct: 20
  doublet_removal: true
annotation:
  method: popv
  reference: tabula_sapiens  # or path to custom h5ad
  n_samples_per_label: 300
malignant:
  tools: [scMalignantFinder, SCEVAN]
  strategy: intersection
  scevan_ref_max_cells: 500
deg:
  test: wilcoxon
  log2fc_threshold: 1.0
  pval_adj_threshold: 0.05
scoring:
  modes: [single, two_gene]
  gates: [AND, OR, NOT]
  safety_threshold: 0.9
  ga_pop_size: 1000
  ga_max_generations: 100
  ga_n_runs: 10
"""
    st.code(yaml_config, language="yaml")
    st.download_button(
        "⬇️ Download YAML config",
        data=yaml_config.encode(),
        file_name="SCART_config.yaml",
        mime="text/yaml",
    )
