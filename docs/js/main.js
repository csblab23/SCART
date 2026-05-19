/* SCART — main.js */

// ── Navbar scroll shadow ──────────────────────────────────────────
const navbar = document.getElementById('navbar');
if (navbar) {
  window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 8);
  }, { passive: true });
}

// ── Copy buttons ──────────────────────────────────────────────────
function copyCode(btn) {
  const pre = btn.closest('.code-wrap, .sbatch-box, .install-strip')
                 ?.querySelector('pre, code');
  if (!pre) return;
  navigator.clipboard.writeText(pre.innerText.trim()).then(() => {
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(() => {
      btn.textContent = orig;
      btn.classList.remove('copied');
    }, 2000);
  });
}

// ── Install command copy ──────────────────────────────────────────
function copyInstall(btn) {
  const cmd = btn.closest('.install-cmd')?.querySelector('code');
  if (!cmd) return;
  navigator.clipboard.writeText(cmd.innerText.trim()).then(() => {
    const orig = btn.textContent;
    btn.textContent = '✓ Copied';
    setTimeout(() => { btn.textContent = orig; }, 2000);
  });
}

// ── Hero code tabs ────────────────────────────────────────────────
document.querySelectorAll('.hero-code-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    const parent = tab.closest('.hero-code');
    parent.querySelectorAll('.hero-code-tab').forEach(t => t.classList.remove('active'));
    parent.querySelectorAll('.hero-code-panel').forEach(p => p.hidden = true);
    tab.classList.add('active');
    const target = parent.querySelector(`[data-panel="${tab.dataset.tab}"]`);
    if (target) target.hidden = false;
  });
});

// ── Sidebar active-link highlight on scroll ───────────────────────
(function () {
  const links = document.querySelectorAll('.sidebar-nav a[href^="#"]');
  if (!links.length) return;

  const heads = Array.from(links).map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);

  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        links.forEach(l => l.classList.remove('active'));
        const active = document.querySelector(`.sidebar-nav a[href="#${entry.target.id}"]`);
        if (active) active.classList.add('active');
      }
    });
  }, { rootMargin: '-70px 0px -60% 0px', threshold: 0 });

  heads.forEach(h => observer.observe(h));
})();

// ── Animate hero elements on load ────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  const animEls = document.querySelectorAll('.hero-badge, .hero h1, .hero-sub, .hero-institution, .hero-ctas, .hero-code');
  animEls.forEach((el, i) => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(18px)';
    el.style.transition = `opacity .55s ease ${i * 0.08}s, transform .55s ease ${i * 0.08}s`;
    setTimeout(() => {
      el.style.opacity = '1';
      el.style.transform = 'translateY(0)';
    }, 60);
  });
});
