(() => {
  const config = window.OMNIFABRIC || {};
  const links = config.links || {};

  function setLink(id, href) {
    const el = document.getElementById(id);
    if (!el) return;
    if (href) {
      el.href = href;
      el.classList.remove('is-disabled');
      el.removeAttribute('aria-disabled');
    } else {
      el.href = '#';
      el.classList.add('is-disabled');
      el.setAttribute('aria-disabled', 'true');
    }
  }

  setLink('paperLink', links.paper);
  setLink('videoLink', links.video);
  setLink('codeLink', links.code);

  const toast = document.getElementById('toast');
  let toastTimer;
  function showToast(message) {
    clearTimeout(toastTimer);
    toast.textContent = message;
    toast.classList.add('is-visible');
    toastTimer = setTimeout(() => toast.classList.remove('is-visible'), 1700);
  }

  async function copyBibtex() {
    try {
      await navigator.clipboard.writeText(config.bibtex || '');
      showToast('BibTeX copied');
    } catch {
      showToast('Copy failed — use the paper section');
    }
  }

  document.getElementById('copyBibtex')?.addEventListener('click', copyBibtex);
  document.getElementById('copyBibtexBottom')?.addEventListener('click', copyBibtex);

  const bibtexBlock = document.getElementById('bibtexBlock');
  if (bibtexBlock) bibtexBlock.textContent = config.bibtex || '';

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const video = document.getElementById('heroVideo');
  if (reducedMotion && video) video.pause();

  if ('IntersectionObserver' in window && !reducedMotion) {
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        }
      }
    }, { threshold: 0.12 });
    document.querySelectorAll('.reveal').forEach((el) => observer.observe(el));
  } else {
    document.querySelectorAll('.reveal').forEach((el) => el.classList.add('is-visible'));
  }
})();
