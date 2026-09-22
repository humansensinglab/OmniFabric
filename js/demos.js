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

  // A flat, non-rotatable 2D viewer for sewing-pattern PNGs: scroll/pinch
  // to zoom, drag to pan, clamped so the pattern can't be dragged away
  // entirely. There is deliberately no rotation control here.
  function createPatternViewer(src, title, onInteract) {
    const MIN_SCALE = 1;
    const MAX_SCALE = 4;

    const wrap = document.createElement('div');
    wrap.className = 'catalog-pattern';
    wrap.hidden = true;

    const img = document.createElement('img');
    img.className = 'catalog-pattern__image';
    img.src = src;
    img.alt = `${title}, sewing pattern`;
    img.draggable = false;
    img.loading = 'lazy';
    img.decoding = 'async';
    wrap.appendChild(img);

    let scale = 1;
    let x = 0;
    let y = 0;

    function clampPan(rect) {
      const maxX = (rect.width * (scale - 1)) / 2;
      const maxY = (rect.height * (scale - 1)) / 2;
      x = Math.min(maxX, Math.max(-maxX, x));
      y = Math.min(maxY, Math.max(-maxY, y));
    }

    function apply() {
      img.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
    }

    function reset(animate) {
      scale = 1;
      x = 0;
      y = 0;
      if (animate) {
        img.style.transition = 'transform .35s cubic-bezier(.2,.8,.2,1)';
        window.setTimeout(() => { img.style.transition = ''; }, 360);
      }
      apply();
    }

    wrap.addEventListener('wheel', (event) => {
      event.preventDefault();
      if (typeof onInteract === 'function') onInteract();
      const rect = wrap.getBoundingClientRect();
      const originX = event.clientX - rect.left - rect.width / 2;
      const originY = event.clientY - rect.top - rect.height / 2;
      const prevScale = scale;
      const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
      scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale * factor));
      const appliedFactor = scale / prevScale;
      x = originX - (originX - x) * appliedFactor;
      y = originY - (originY - y) * appliedFactor;
      if (scale === MIN_SCALE) { x = 0; y = 0; }
      clampPan(rect);
      apply();
    }, { passive: false });

    const pointers = new Map();
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    let pinchDist = 0;
    let pinchScale = 1;

    wrap.addEventListener('pointerdown', (event) => {
      if (typeof onInteract === 'function') onInteract();
      wrap.setPointerCapture(event.pointerId);
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      if (pointers.size === 1) {
        dragging = true;
        lastX = event.clientX;
        lastY = event.clientY;
      } else if (pointers.size === 2) {
        dragging = false;
        const pts = [...pointers.values()];
        pinchDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
        pinchScale = scale;
      }
    });

    wrap.addEventListener('pointermove', (event) => {
      if (!pointers.has(event.pointerId)) return;
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      const rect = wrap.getBoundingClientRect();

      if (pointers.size === 2) {
        const pts = [...pointers.values()];
        const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
        if (pinchDist > 0) {
          scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, pinchScale * (dist / pinchDist)));
          clampPan(rect);
          apply();
        }
        return;
      }

      if (dragging && scale > MIN_SCALE) {
        x += event.clientX - lastX;
        y += event.clientY - lastY;
        lastX = event.clientX;
        lastY = event.clientY;
        clampPan(rect);
        apply();
      }
    });

    function endPointer(event) {
      pointers.delete(event.pointerId);
      if (pointers.size < 2) pinchDist = 0;
      if (pointers.size === 0) dragging = false;
    }
    wrap.addEventListener('pointerup', endPointer);
    wrap.addEventListener('pointercancel', endPointer);
    wrap.addEventListener('pointerleave', endPointer);

    wrap.addEventListener('dblclick', () => reset(true));

    return { element: wrap, reset: () => reset(true) };
  }

  // allowPattern gates the 3D/Pattern toggle at the call site, not just
  // via data: Composed Outfit cards never get it, even if an outfit
  // entry in js/data.js ever ends up with a stray "pattern" field.
  function createViewer(item, { allowPattern = false } = {}) {
    const panel = document.createElement('div');
    panel.className = 'catalog-viewer';

    const meta = document.createElement('div');
    meta.className = 'catalog-viewer__meta';

    const title = document.createElement('span');
    title.textContent = `${item.id} — ${item.title}`;

    // Only created for single-garment cards, and only shown once a
    // matching sewing-pattern asset is confirmed to exist (see the
    // item.pattern check near the end of this function).
    let modeGroup = null;
    let modeButtons = [];
    if (allowPattern) {
      modeGroup = document.createElement('div');
      modeGroup.className = 'catalog-viewer__mode';
      modeGroup.setAttribute('role', 'group');
      modeGroup.setAttribute('aria-label', 'View mode');
      modeGroup.hidden = true;

      modeButtons = ['3d', 'pattern'].map((mode) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'catalog-viewer__mode-btn';
        btn.dataset.mode = mode;
        btn.textContent = mode === '3d' ? '3D' : 'Pattern';
        btn.setAttribute('aria-pressed', String(mode === '3d'));
        modeGroup.appendChild(btn);
        return btn;
      });
    }

    const reset = document.createElement('button');
    reset.type = 'button';
    reset.className = 'text-link catalog-viewer__reset';
    reset.textContent = 'Reset ↺';
    meta.appendChild(title);
    if (modeGroup) meta.appendChild(modeGroup);
    meta.appendChild(reset);

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

    function setHintForMode(mode) {
      hint.innerHTML = mode === 'pattern'
        ? '<span>⤢</span> Scroll to zoom · drag to pan'
        : '<span>⟲</span> Drag to rotate';
      hint.classList.remove('is-hidden');
    }

    function onCameraChange(event) {
      if (event.detail && event.detail.source === 'user-interaction') {
        hideHint();
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

    // ---- 3D / Pattern mode switching (single-garment examples only) ----
    let currentMode = '3d';
    let patternController = null;

    function activateMode(mode) {
      if (mode === currentMode || (mode === 'pattern' && !patternController)) return;
      currentMode = mode;
      modeButtons.forEach((btn) => {
        btn.setAttribute('aria-pressed', String(btn.dataset.mode === mode));
      });
      if (mode === 'pattern') {
        model.style.visibility = 'hidden';
        model.style.pointerEvents = 'none';
        patternController.element.hidden = false;
      } else {
        patternController.element.hidden = true;
        if (item.model) {
          model.style.visibility = '';
          model.style.pointerEvents = '';
        }
      }
      setHintForMode(mode);
    }

    modeButtons.forEach((btn) => {
      btn.addEventListener('click', () => activateMode(btn.dataset.mode));
    });

    reset.addEventListener('click', () => {
      if (currentMode === 'pattern' && patternController) patternController.reset();
      else resetModelView(model);
    });

    stage.append(model, empty, hint);
    panel.append(meta, stage);

    if (allowPattern && item.pattern) {
      assetExists(item.pattern).then((ok) => {
        if (!ok) return;
        patternController = createPatternViewer(item.pattern, item.title, hideHint);
        stage.appendChild(patternController.element);
        modeGroup.hidden = false;
      });
    }

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

      body.append(inputWrap, createViewer(item, { allowPattern: true }));
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
