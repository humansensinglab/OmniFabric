# OmniFabric project page starter

A no-build static project page designed for GitHub Pages.

## What is already implemented

- full-screen teaser video hero
- BOLERO-inspired white editorial masthead positioned in the teaser's top title zone
- ORBIT/Cloister-inspired ivory + blue visual system
- responsive layout
- introduction / method / interactive / video / paper sections
- two interactive demo groups
- one live `<model-viewer>` per group (models swap instead of loading 12 viewers at once)
- keyboard-accessible numbered demo selector
- BibTeX copy buttons
- reduced-motion support
- mobile layouts
- placeholder behavior when assets are still missing

`<model-viewer>` is pinned to version 4.3.1 for reproducibility.

## Files the coding agent normally needs to edit

### `js/data.js`
The content manifest. Put your final image and GLB paths here.

### `index.html`
Edit prose, section content, author links, and add/remove optional sections.

### `css/styles.css`
Fine-tune typography, spacing, exact colors, hero overlay geometry, and mobile behavior.

The agent should **not** need to redesign the project architecture.

## Preview locally

From the repository root:

```bash
python3 -m http.server 8000
```

Open:

```text
http://localhost:8000
```

Do not preview by double-clicking `index.html` if you are testing GLBs; browser file-origin restrictions can behave differently from HTTP/GitHub Pages.

## GitHub Pages deployment

1. Commit these files to the repository's `main` branch.
2. On GitHub, open **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Select `main` and `/ (root)`.
5. Save.

For a repository named `<username>.github.io`, the site is served at the account root. For another repo name, GitHub Pages serves it under `/<repo-name>/`; this starter uses relative asset paths so both work.

## Before giving the repository to the coding agent

Read `ASSET_PREP_CHECKLIST.md`.

You can give the agent the repo even before all assets exist. The best workflow is:

1. put in the teaser video
2. put in the 12 reference-image sets
3. put in as many optimized GLBs as are ready
4. edit `js/data.js` with those paths
5. give the agent the design-plan DOCX + this repository
6. ask the agent to tune placement/typography against the real media

## Suggested final coding-agent task

> The functional starter site is already implemented. Do not rewrite it from scratch. Use the supplied design plan and real assets to fine-tune the page. Prioritize: exact BOLERO-like hero masthead placement over the teaser video; ORBIT/Cloister-style editorial spacing, ivory/blue typography and section composition; accurate responsive behavior; and polished rendering of the 12 interactive GLB examples. Keep the UI minimal. Modify `js/data.js` for asset paths rather than hard-coding demo content throughout the HTML.
