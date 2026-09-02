# Heatwave Intelligence Platform

**Use Case KJS-CES-01 — Climate Intelligence for Heatwave Monitoring, Prediction and Early Warning**

A static, dependency-free decision-support website that implements the five-phase conceptual schema
of use case KJS-CES-01: region-wise heatwave forecasting from India Meteorological Department
observations, IoT-enabled Automated Weather Station monitoring, forecast validation, and
LLM-assisted advisory generation for citizens, farmers, health agencies and local authorities.

**Live site:** https://jiviteshkumar.github.io/climate_/

---

## Pages

| Page | Purpose |
| --- | --- |
| `index.html` | Project overview, problem framing, five-phase solution approach, current outlook |
| `dashboard.html` | Heatwave Watch — region-wise severity, hotspot map, exposed population |
| `forecast.html` | AI forecasting models, seasonal heatmap, warming trends, model evaluation |
| `monitoring.html` | IoT Automated Weather Station telemetry and forecast validation |
| `advisories.html` | Stakeholder-specific advisories and dissemination matrix |
| `about.html` | Use case background, objectives, governance and technology stack |
| `404.html` | Not-found page |

## Structure

```
climate_/
├── index.html
├── dashboard.html
├── forecast.html
├── monitoring.html
├── advisories.html
├── about.html
├── 404.html
├── robots.txt
├── sitemap.xml
└── assets/
    ├── css/style.css
    ├── js/data.js        # region, seasonal, forecast, AWS and advisory reference data
    ├── js/main.js        # SVG chart engine and page logic
    └── img/              # favicon and social cover
```

## Technology

- HTML5, CSS3 and vanilla JavaScript — no frameworks, no build step, no external runtime dependency
- Every chart, heatmap and the schematic regional map are rendered as hand-built inline SVG
- Responsive down to 360 px, keyboard navigable, with `prefers-reduced-motion` respected
- Hosted on GitHub Pages from the `main` branch

## Search engine optimization

Each page carries a unique title tag, meta description, meta keywords, meta robots directive,
canonical URL, Open Graph and Twitter card metadata, a single `H1` with an ordered heading
hierarchy, descriptive anchor text with `title` attributes, and `aria-label` text on every inline
SVG graphic. Crawl directives are published in `robots.txt`, which references `sitemap.xml`.

## Web analytics

Google Analytics 4 is wired into every page via the `gtag.js` tag in the document head. Replace the
placeholder `G-XXXXXXXXXX` with the Measurement ID of your own GA4 data stream:

```bash
grep -rl "G-XXXXXXXXXX" . --include=*.html | xargs sed -i "s/G-XXXXXXXXXX/G-YOURID/g"
```

## Data

Historical maximum-temperature observations come from the India Meteorological Department, Pune
gridded archive: https://imdpune.gov.in/lrfindex.php

The values bundled in `assets/js/data.js` are representative climatological figures used to drive
this demonstration interface. A production deployment reads the live IMD archive and the live AWS
ingest stream in their place.

## Attribution

Built by **Jivitesh Kumar** (Roll No. 16010423041, Batch A2), B.Tech Information Technology,
Semester VII, Department of Information Technology, K J Somaiya School of Engineering, Somaiya
Vidyavihar University — Digital Marketing laboratory, academic year 2026–27.

This is an academic demonstration platform. Operational heatwave warnings for India are issued
only by the India Meteorological Department.
