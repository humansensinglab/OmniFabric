(() => {
  const config = window.OMNIFABRIC || {};

  function padNumber(value) {
    return String(value).padStart(2, '0');
  }

  // An example is only shown once every asset it needs (all reference
  // images plus the GLB) actually resolves in the repo. This lets
  // js/data.js keep listing every planned slot without the page showing
  // broken/placeholder cards for ones not finished yet.
  async function assetExists(url) {
    if (!url) return false;
    try {
      const response = await fetch(url, { method: 'HEAD', cache: 'no-store' });
      return response.ok;
    } catch {
      return false;
    }
  }

  async function isExampleReady(item) {
    const paths = [...(item.inputs || []), item.model];
    if (!paths.length || paths.some((path) => !path)) return false;
    const results = await Promise.all(paths.map(assetExists));
    return results.every(Boolean);
  }

  async function filterReadyExamples(examples) {
    const flags = await Promise.all(examples.map(isExampleReady));
    return examples.filter((_, index) => flags[index]);
  }

  function createReferenceCard(src, index, total) {
    const figure = document.createElement('figure');
    figure.className = 'catalog-reference';

    if (src) {
      const img = document.createElement('img');
      img.className = 'catalog-reference__image';
      img.src = src;
      img.alt = total > 1 ? `Input reference ${index + 1}` : 'Input garment reference';
      img.loading = 'lazy';
      img.decoding = 'async';
      figure.appendChild(img);
    } else {
      const empty = document.createElement('div');
      empty.className = 'catalog-reference__empty';
      empty.textContent = total > 1 ? `Input ${index + 1}` : 'Input image';
      figure.appendChild(empty);
    }

    const caption = document.createElement('figcaption');
    caption.innerHTML = `<span>${total > 1 ? `Reference ${padNumber(index + 1)}` : 'Reference'}</span><span>Input</span>`;
    figure.appendChild(caption);
    return figure;
  }

  function resetModelView(model) {
    if (!model || !model.src) return;
    model.cameraOrbit = '0deg 75deg auto';
    model.cameraTarget = 'auto auto auto';
    model.fieldOfView = 'auto';
    if (typeof model.jumpCameraToGoal === 'function') model.jumpCameraToGoal();
  }

  function createViewer(item) {
    const panel = document.createElement('div');
    panel.className = 'catalog-viewer';

    const meta = document.createElement('div');
    meta.className = 'catalog-viewer__meta';

    const title = document.createElement('span');
    title.textContent = `${item.id} — ${item.title}`;

    const reset = document.createElement('button');
    reset.type = 'button';
    reset.className = 'text-link catalog-viewer__reset';
    reset.textContent = 'Reset ↺';
    meta.append(title, reset);

    const stage = document.createElement('div');
    stage.className = 'catalog-viewer__stage';

    const model = document.createElement('model-viewer');
    model.className = 'garment-viewer garment-viewer--catalog';
    model.setAttribute('camera-controls', '');
    model.setAttribute('touch-action', 'pan-y');
    model.setAttribute('interaction-prompt', 'none');
    model.setAttribute('auto-rotate', '');
    model.setAttribute('auto-rotate-delay', '0');
    model.setAttribute('rotation-per-second', '6deg');
    model.setAttribute('shadow-intensity', '0.45');
    model.setAttribute('shadow-softness', '0.9');
    model.setAttribute('exposure', '0.95');
    model.setAttribute('tone-mapping', 'neutral');
    model.setAttribute('loading', 'lazy');
    model.setAttribute('reveal', 'auto');
    model.alt = `${item.title}, interactive 3D reconstruction`;

    const empty = document.createElement('div');
    empty.className = 'viewer-empty viewer-empty--catalog';
    empty.innerHTML = '<span>3D MODEL</span>';

    const hint = document.createElement('div');
    hint.className = 'viewer-hint viewer-hint--catalog';
    hint.setAttribute('aria-hidden', 'true');
    hint.innerHTML = '<span>⟲</span> Drag to rotate';

    function hideHint() {
      hint.classList.add('is-hidden');
    }

    function onCameraChange(event) {
      if (event.detail && event.detail.source === 'user-interaction') {
        hideHint();
        model.removeEventListener('camera-change', onCameraChange);
      }
    }

    if (item.model) {
      model.src = item.model;
      empty.hidden = true;
      empty.setAttribute('aria-hidden', 'true');
      model.addEventListener('camera-change', onCameraChange);
    } else {
      model.style.visibility = 'hidden';
      model.style.pointerEvents = 'none';
      empty.hidden = false;
      empty.setAttribute('aria-hidden', 'false');
      hideHint();
    }

    model.addEventListener('error', () => {
      empty.hidden = false;
      empty.setAttribute('aria-hidden', 'false');
      empty.innerHTML = '<span>MODEL UNAVAILABLE</span>';
      hideHint();
    });

    reset.addEventListener('click', () => resetModelView(model));
    stage.append(model, empty, hint);
    panel.append(meta, stage);
    return panel;
  }

  function createIndexLine(item, index) {
    const line = document.createElement('div');
    line.className = 'catalog-index';
    line.innerHTML = `<span>${padNumber(index + 1)}</span><span>${item.category || 'Garment'}</span>`;
    return line;
  }

  function renderSingles(root, examples) {
    examples.forEach((item, index) => {
      const article = document.createElement('article');
      article.className = 'single-demo-pair';

      const body = document.createElement('div');
      body.className = 'single-demo-pair__body';

      const inputWrap = document.createElement('div');
      inputWrap.className = 'single-demo-pair__input';
      const src = (item.inputs || [])[0] || '';
      inputWrap.appendChild(createReferenceCard(src, 0, 1));

      body.append(inputWrap, createViewer(item));
      article.append(createIndexLine(item, index), body);
      root.appendChild(article);
    });
  }

  function renderOutfits(root, examples) {
    examples.forEach((item, index) => {
      const article = document.createElement('article');
      article.className = 'outfit-demo-pair';

      const body = document.createElement('div');
      body.className = 'outfit-demo-pair__body';

      const inputs = document.createElement('div');
      inputs.className = 'outfit-demo-pair__inputs';
      const refs = item.inputs || [];
      for (let i = 0; i < 2; i += 1) {
        inputs.appendChild(createReferenceCard(refs[i] || '', i, 2));
      }

      body.append(inputs, createViewer(item));
      article.append(createIndexLine(item, index), body);
      root.appendChild(article);
    });
  }

  document.querySelectorAll('[data-demo-grid]').forEach(async (root) => {
    const key = root.dataset.demoGrid;
    const examples = config[key] || [];
    const section = root.closest('.gallery-section');
    const ready = examples.length ? await filterReadyExamples(examples) : [];

    if (!ready.length) {
      if (section) section.hidden = true;
      return;
    }

    if (key === 'single') renderSingles(root, ready);
    if (key === 'outfit') renderOutfits(root, ready);
  });
})();
