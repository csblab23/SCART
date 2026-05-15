// ── CODE SNIPPETS FOR USAGE SECTION ─────────────────────────────────
const codeSnippets = [
  {
    lang: "bash",
    code: `# Install SCART
pip install git+https://github.com/Vinaya20000/SCART.git

# Or clone and install locally
git clone https://github.com/Vinaya20000/SCART.git
cd SCART
pip install -e .`
  },
  {
    lang: "python",
    code: `from SCART.geo_fetcher import SampleAnnotator

# Fetch GEO datasets with QC
annotator = SampleAnnotator(
    "GSE162499", "GSE144735",
    cancer_type="lung_cancer",
    min_genes=200,
    max_mt=20
)
normal, tumor, unspecified, ann, h5ad, ct, results = annotator.run()
print(f"Tumor h5ad saved: {h5ad}")`
  },
  {
    lang: "python",
    code: `from SCART.popv_annotation import auto_run_popv

# Run PopV consensus annotation
adata = auto_run_popv(
    nsamples=300,
    output_dir="popv_results",
    # user_reference="path/to/Lung_TSP1_30.h5ad"  # optional
)
print(adata.obs["popv_majority_vote_prediction"].value_counts())`
  },
  {
    lang: "python",
    code: `from SCART.gene_combination_predictor.one_gene_combination import run as run_single
from SCART.gene_combination_predictor.two_gene_combination import run as run_two

# Single gene scoring
df_single = run_single(safety_threshold=0.9)

# Two-gene GA scoring (AND/OR/NOT gates)
df_hof, df_all = run_two(
    safety_threshold=0.9,
    pop_size=1000,
    n_runs=10
)
print(df_hof.head(10))`
  }
];

function showCode(idx, el) {
  document.querySelectorAll(".usage-step").forEach(s => s.classList.remove("active"));
  el.classList.add("active");
  const snippet = codeSnippets[idx];
  document.getElementById("code-display").innerHTML =
    `<code>${escapeHtml(snippet.code)}</code>`;
  document.getElementById("code-lang").textContent = snippet.lang;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function copyCode() {
  const code = document.getElementById("code-display").innerText;
  navigator.clipboard.writeText(code).then(() => {
    const btn = document.getElementById("copy-btn");
    btn.textContent = "Copied!";
    setTimeout(() => btn.textContent = "Copy", 2000);
  });
}

function copyCitation() {
  const bibtex = document.getElementById("bibtex");
  bibtex.style.display = bibtex.style.display === "none" ? "block" : "none";
  if (bibtex.style.display === "block") {
    const text = bibtex.innerText;
    navigator.clipboard.writeText(text).catch(() => {});
  }
}

// ── NAVBAR SCROLL EFFECT ─────────────────────────────────────────────
window.addEventListener("scroll", () => {
  const nav = document.getElementById("navbar");
  if (window.scrollY > 20) {
    nav.style.background = "rgba(5,8,16,0.97)";
  } else {
    nav.style.background = "rgba(5,8,16,0.85)";
  }
});

// ── HAMBURGER MENU ───────────────────────────────────────────────────
document.getElementById("hamburger").addEventListener("click", () => {
  const links = document.querySelector(".nav-links");
  if (links.style.display === "flex") {
    links.style.display = "none";
  } else {
    links.style.cssText = `
      display: flex; flex-direction: column; position: fixed;
      top: 64px; left: 0; right: 0;
      background: rgba(5,8,16,0.98); padding: 20px 32px;
      border-bottom: 1px solid rgba(99,168,255,0.1);
      gap: 16px; z-index: 999;
    `;
  }
});

// ── FADE-IN ANIMATION ON SCROLL ──────────────────────────────────────
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = "1";
      entry.target.style.transform = "translateY(0)";
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll(".pipe-step, .gate-card, .db-card").forEach(el => {
  el.style.opacity = "0";
  el.style.transform = "translateY(20px)";
  el.style.transition = "opacity 0.5s ease, transform 0.5s ease";
  observer.observe(el);
});
