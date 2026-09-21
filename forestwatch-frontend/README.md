# ForestWatch India — React Frontend

React + Vite replacement for the original single-file `index.html`.
Talks to the existing FastAPI backend (`main.py`) — nothing on the
backend needs to change to use this.

## Setup

```bash
npm install
npm run dev
```

Opens at `http://localhost:5173`. Make sure your FastAPI backend is
running separately (`python -m uvicorn main:app --reload`) on
`http://127.0.0.1:8000` — that's the default the frontend expects.

If your backend runs somewhere else, copy `.env.example` to `.env` and
change `VITE_API_URL`.

## Build for deployment

```bash
npm run build
```

Outputs static files to `dist/` — deploy that folder to any static host
(Netlify, Vercel, GitHub Pages, etc.). Remember to update `VITE_API_URL`
to your deployed backend's URL before building, and update the backend's
CORS `allow_origins` in `main.py` away from `"*"` to your deployed
frontend's actual origin.

## Project structure

```
src/
  api.js                 -- all backend calls in one place
  styles.css              -- design tokens + global styles (ported from
                              the original index.html's editorial look)
  App.jsx                 -- state management, mode switching
  components/
    Masthead.jsx
    Controls.jsx           -- city/mode/year selectors, run button
    Stage.jsx               -- main visual area for all 6 modes
    SidePanel.jsx           -- routes to the correct stats panel
    FragmentationGrid.jsx   -- shared by single-year + prediction panels
    Footer.jsx
    panels/
      SinglePanel.jsx
      ComparePanel.jsx
      PredictPanel.jsx
      RiskPanel.jsx
      ConservationPanel.jsx
      HansenPanel.jsx
```

## Modes

- **Single year** — satellite/overlay/mask toggle, carbon stats,
  fragmentation metrics, historical trend bars
- **Compare years** — side-by-side + diff map, change breakdown
- **Project forward** — CA-ANN projection, validation summary,
  fragmentation trend (base year vs. projected year)
- **Risk heatmap** — CA-ANN's continuous per-pixel risk score
- **Conservation priority** — top-10 ranked patches, numbered markers on
  the map matching the ranked list
- **Hansen validation** — strict + buffered accuracy against Hansen GFC,
  with the full methodology notes from the backend
