# OmniFabric asset-preparation checklist

Prepare these before asking the coding agent to do the final polish. You do **not** need everything before the first coding pass; the starter page works with placeholders.

## 1. Hero assets — highest priority

Place these in `assets/hero/`:

- `omnifabric_teaser.mp4`
  - H.264 MP4 is the safest web format.
  - 1920×1080 is enough for desktop; do not upload an enormous lossless file.
  - Remove audio unless the sound is intentionally part of the page.
  - Make the clip loop cleanly.
  - Target roughly 5–20 MB if practical.
- `hero-poster.png`
  - Already filled with the sample frame you provided.
  - Replace later with the exact desired poster frame if needed.

**Important:** preserve the empty title region at the top of the teaser composition. The HTML overlay is intentionally positioned there.

## 2. Paper / project links

Already included:

- `assets/paper/omnifabric.pdf`

Before launch, confirm/update in `js/data.js`:

- paper link
- project video link
- code link
- BibTeX

## 3. The 12 interactive examples

Decide which are:

- **Single-Garment Reconstruction**
- **Composed Outfit Reconstruction** (two input images: upper + lower garment)

The starter currently allocates **6 slots to each group**. If your real split is different, just add/remove entries in `js/data.js`; no layout rewrite is required.

For each example prepare:

### Single garment

- one reference image, preferably `.webp` or `.jpg`
- one optimized `.glb`

Suggested names:

- `assets/images/singles/s01.webp`
- `assets/models/singles/s01.glb`

### Composed outfit

- upper garment reference image
- lower garment reference image
- one optimized `.glb`

Suggested names:

- `assets/images/outfits/o01-top.webp`
- `assets/images/outfits/o01-bottom.webp`
- `assets/models/outfits/o01.glb`

## 4. GLB preparation guidelines

Use GLB as the browser-delivery format. Keep `.blend`, `.fbx`, `.abc`, `.zprj` as master/interchange files only.

For each web GLB:

- include avatar + garment in the same scene if they must remain aligned
- preserve UVs and textures
- use real-time/PBR-compatible materials
- remove hidden/unnecessary geometry
- reduce extreme simulation mesh density if possible
- resize textures to 1K–2K where visual quality allows
- aim for ~5–20 MB per model when practical
- test the GLB in model-viewer before committing all assets

Do **not** use Git LFS for files that GitHub Pages needs to serve directly.

## 5. Method / explanation image

Optional but recommended:

- `assets/images/figures/method.webp`

A clean landscape figure is ideal. The current page has a placeholder block for it.

## 6. Extra motion clips

Optional:

- `assets/videos/highlight-01.mp4`
- `assets/videos/highlight-02.mp4`

Use these for CLO/Blender motion, fabric movement, or qualitative showcases.

## 7. Author links / exact metadata

Confirm:

- exact author spelling/order
- author homepage links
- venue wording
- project title capitalization
- paper/code/video URLs

## 8. Before final deployment

- test desktop Chrome/Safari/Firefox
- test an iPhone Safari viewport
- verify all GLBs load over HTTPS
- verify video autoplay (must be muted + playsinline)
- compress oversized images
- check total repository/site size
- inspect page with slow-network throttling
- verify links/BibTeX
- run Lighthouse once for obvious performance/accessibility problems
