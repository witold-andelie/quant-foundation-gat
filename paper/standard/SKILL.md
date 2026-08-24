---
name: wh-hs-report
description: >-
  Personal template and W-HS (Westphalian University of Applied Sciences)
  identity for Wentao Ma's course reports written in the ACM acmart LaTeX
  template. Use whenever writing, scaffolding, or formatting a technical
  report / seminar report / term paper for a W-HS course (e.g. ECCR), or when
  the user references "the report template", "my w-hs info", or "the acmart
  course template". Provides the fixed author/course metadata and the exact
  acmart preamble tweaks (margins, page style, centered title) to reuse.
---

# W-HS course report template (Wentao Ma)

Use this when producing a course report for Wentao Ma at Westphalian
University of Applied Sciences based on the provided `acmart` template
(`main_wh_acmart.tex` + `wh.sty` + `01_Sections/` + `02_Figures_Tables/`).

This skill lives with the template on the D drive (inside the project folder,
under `.claude/skills/`), so it travels whenever the template folder is copied
to start a new report. Do not move it back to the C-drive user skills folder.

## Fixed author / institution metadata (do not re-ask)

| Field | Value |
|---|---|
| Author | Wentao Ma |
| Student email | ma.wentao@studmail.w-hs.de |
| Institution | Westphalian University of Applied Sciences |
| Department | Computer Science and Communication |
| Cooperation | Institute for Internet Security -- if(is) |
| Personal email | andelie1892@gmail.com |

Course metadata varies per report — confirm each time (course name, supervisor,
semester). Known so far: course **Emerging Challenges in Cybersecurity
Research (ECCR)**, supervisor **Prof. Tobias Urban**, **summer semester 2026**.

## Report conventions (from the ECCR technical report)

- Class: `\documentclass[sigconf, 10pt, ..., manuscript, nonacm]{acmart}`
  (single column, 10pt).
- A **Preface** section states course, supervisor, semester, and the
  completion date (use the date the user gives, not today's date).
- Acknowledgments **must** contain a generative-AI usage disclosure (course
  rule): state what AI did, that the author verified all claims/numbers, and
  that the author is responsible for the final text.
- Every quantitative claim traces to a source artifact; supporting citations
  are taken from the source papers' own reference lists, verified, never from
  memory.
- Figures reproduced from source papers are captioned "Figure from~\cite{...}"
  and given an `\Description{}` for accessibility.

## Required acmart preamble tweaks (course wants these)

acmart's manuscript defaults look wrong for this course; apply these fixes:

```latex
% 1. Smaller, symmetric margins (fuller page; also kills the odd/even
%    left-right alternation that acmart's twoside binding margins cause,
%    which the `oneside' class option does NOT fix). geometry is already
%    loaded by acmart, so call \geometry, do not \usepackage it again.
\geometry{left=1in,right=1in,top=0.9in,bottom=1in}

% 2. Center the (already \sffamily\bfseries) title and add space below it.
\usepackage{etoolbox}
\makeatletter
\patchcmd{\@mktitle@i}{\raggedright}{\centering}{}{}
\patchcmd{\@mktitle@i}{\par\bigskip}{\par\bigskip\vspace{6pt}}{}{}
\makeatother
```

```latex
% 3. Page number bottom-center, no running header/author byline in the corner.
%    Put this right AFTER \maketitle in the body:
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0pt}
```

## Build

MiKTeX at `D:\MiKTeX\miktex\bin\x64\`. Build:
`pdflatex -> bibtex -> pdflatex -> pdflatex`. If `main_*.pdf` is locked
(open in a viewer) the build fails with "I can't write on file"; build to a
temporary `-jobname` to preview, and rebuild the real file once the viewer is
closed. Verify with the claude-latex-paper-skill's `verify_paper.py`.
