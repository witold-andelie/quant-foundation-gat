# Design Document

## 1. Profile Baseline Declaration

- **Profile selection**: `profiles/academic.md`
- **Selection rationale**: Master's capstone presentation to a professor (Prof. Anderle) with CS and biology background. Academic style with emphasis on clear methodology, figures, and honest reporting of both positive and negative results.
- **Referenced dimensions**: Information density (high), chart-dominant text-to-visual ratio, navigation bar, clean color scheme, argumentation-driven narrative, original figure reuse.
- **Deviation notes**: 
  - The audience is not from finance; alpha concepts need extra explanation (plain-language primer).
  - The reference PPT uses a white background with teal/blue header accent; we follow this rather than the generic academic profile's potentially darker schemes.
  - We include a bottom navigation bar (page number + project title) as seen in the reference PPT.

## 2. Style Baseline Declaration

- **Style anchor identification**: Swiss International Style + clean tech-academic hybrid. The reference PPT (GAT_Capstone_Report) uses flat solid colors, strong left-aligned hierarchy, grid-based layout, and minimal decoration.
- **Referenced dimension explanation**: We reference the reference PPT's layout system (header band, footer band, card-based content blocks) and its color restraint (mostly white + one accent color).
- **Reference scope declaration**: Style + color scheme + layout (excluding content, which is from the final paper).

## 3. Extract Style from Reference Source

### 3.1 Typographic character
- Clean, rational, information-dense but not cluttered. Every slide has a clear top-down reading order: section badge → headline → body → insight bar.

### 3.2 Color Extraction
- **primary**: `#0B5C5E` (deep teal — the header/badge color from the reference PPT, used for titles, navigation, key anchors)
- **secondary**: `#3D8B8E` (lighter teal, used for secondary badges, supporting shapes)
- **accent**: `#E07A5F` (muted terracotta, used sparingly for negative results, warnings, and key highlights that need to pop against the teal)
- **background**: `#FFFFFF` (white — dominant background throughout, matching the reference)
- **text**: `#1A1A2E` (near-black navy, used for body text; softer than pure black for reduced eye strain)
- Additional grays: `#F5F5F7` (light card fill), `#E5E5E8` (dividers), `#6B7280` (footnotes/annotations)

Color usage rules:
- Titles and section badges: $primary
- Body text: $text
- Cards/containers: `#F5F5F7` fill with no border or `#E5E5E8` subtle border
- Key positive findings: $primary or $secondary
- Key negative findings / cautionary notes: $accent
- Bottom navigation bar: $primary fill with white text

### 3.3 Font Hierarchy Extraction
- Reference PPT uses a clean sans-serif (likely Calibri or similar). We use QuattrocentoSans for academic elegance and screen readability.
- **Cover title**: 40px, QuattrocentoSans, bold, $primary
- **Page title**: 28px, QuattrocentoSans, bold, $text
- **Section badge**: 14px, QuattrocentoSans, uppercase, letter-spacing 2px, $primary
- **Subtitle / figure caption**: 18px, QuattrocentoSans, regular, $text
- **Body**: 18px, QuattrocentoSans, regular, $text, line-height 1.5
- **Footnotes / navigation**: 12px, QuattrocentoSans, regular, white (on nav bar) or `#6B7280` (on white bg)
- **Table text**: 14px, QuattrocentoSans, regular
- **Chart labels**: 12px, QuattrocentoSans, regular

### 3.4 Text Box and Container Styles
- **Cards**: rounded rectangles (roundRect, adjustments [3000]), fill `#F5F5F7`, no border or 1px `#E5E5E8`
- **Section badge**: small teal rectangle with white text, or teal text on white with underline
- **Divider lines**: horizontal 1px solid `#E5E5E8` or `#0B5C5E`
- **No heavy shadows, no gradients on containers** — flat, clean academic style

### 3.5 Image Style
- **Icons**: solid (fas), monochrome $primary or $secondary, small (24px), used only for list bullets or card headers
- **Tables**: minimal three-line style; header row $primary fill with white text; alternating body rows white / `#F5F5F7`; no heavy borders
- **Charts**: minimal style, series colors [$primary, $secondary, $accent, `#6B7280`, `#2D9CDB`]; grid lines light gray; no chart shadows
- **Illustrations**: Reuse original paper figures directly (isolation_anchors, interconnector, energy_ladder, attention_weight). They are vector PDFs or PNGs, placed at adequate size. No re-drawing.

## 4. Layout System

### 4.1 Global Layout Characteristics
- **Page size**: 1280 x 720 (16:9)
- **Page margins**: Left 60px, right 60px, top 40px, bottom 50px (content area above footer)
- **Footer bar**: fixed at y=680, height 40px, full-width $primary fill, white text. Left: "Relational Alpha Factors · GAT Capstone"; Right: page number
- **Header area**: y=40 to y=120. Section badge at top, title below it.
- **Content area**: y=130 to y=660
- **No logo** (user did not specify a university logo)

### 4.2 Special Page Layouts
- **Cover**: Full-bleed white background. Title centered horizontally, slightly above center (y=220). Subtitle below. Bottom: author name + date. No footer bar on cover.
- **Table of contents**: White background. Title at top. Content as a clean grid of chapter cards (2 columns, 3 rows), each card with a teal number circle and chapter title.
- **Final page**: White background. Key takeaway text centered, large font. Below: future directions as a small list. Footer bar present.

### 4.3 Content Page Layout Patterns
- **Left-figure, right-text**: 55% image left, 45% text right (or vice versa). Used for figure-heavy slides.
- **Top-badge, bottom-grid**: Section badge + title at top, then 2-3 content cards below in a row. Used for methodology / controls.
- **Full-width table/chart**: Title at top, then a full-width table or chart filling the content area. Used for results summary.
- **Two-column comparison**: Two equal columns with headers, used for equity vs energy track comparison.
- **Text + bottom insight bar**: Body text above, then a teal or gray horizontal bar at the bottom with the key insight in white or bold text. Used for conclusion slides.

## 5. Style Usage Rules

- **$title** textStyle: cover title, chapter divider titles
- **$pageTitle** textStyle: content page main titles
- **$badge** textStyle: section category badges (e.g., "MOTIVATION", "RESULTS")
- **$body** textStyle: all body paragraphs, bullet lists, figure captions
- **$footnote** textStyle: source notes, footnotes, navigation bar text
- **$tableHeader** tableStyle: table header rows
- **$default** tableStyle: standard data tables
- Colors: $primary for titles/badges/navbar; $text for body; $accent for warnings/negative highlights; $background for page bg; `#F5F5F7` for card fills.

## 6. Risk Prohibitions

- [ ] No dark background slides (impairs figure readability for PDF charts)
- [ ] No gradient fills on shapes or backgrounds (academic flat style)
- [ ] No decorative clip-art icons (only data-driven figures and minimal fas icons)
- [ ] No font size below 12px for any readable text; body must stay at 18px minimum
- [ ] No title font size below 26px
- [ ] No flashy animations or transitions (not applicable in PPTD but stated for clarity)
- [ ] No high-saturation neon colors (teal is already strong; avoid red/green traffic-light semantics unless for the accent caution color)
- [ ] No rounded rectangles with very large radius (keep academic sharpness, max 3000)
- [ ] Do not redraw original paper figures; embed them directly from the figures/ folder
- [ ] Do not use finance jargon without brief plain-language translation on the alpha-primer slide
- [ ] No text-only slides exceeding 60% of the content area — always pair with a figure, table, or diagram

## 7. Theme Definition

```yaml
theme:
  colors:
    primary: "#0B5C5E"
    secondary: "#3D8B8E"
    accent: "#E07A5F"
    background: "#FFFFFF"
    text: "#1A1A2E"
    lightGray: "#F5F5F7"
    midGray: "#E5E5E8"
    darkGray: "#6B7280"
  textStyles:
    title:
      fontSize: 40
      color: "$primary"
      fontFamily: "QuattrocentoSans"
      lineHeight: 1.2
    pageTitle:
      fontSize: 28
      color: "$text"
      fontFamily: "QuattrocentoSans"
      lineHeight: 1.2
    badge:
      fontSize: 14
      color: "$primary"
      fontFamily: "QuattrocentoSans"
      letterSpacing: 2
      lineHeight: 1.2
    body:
      fontSize: 18
      color: "$text"
      fontFamily: "QuattrocentoSans"
      lineHeight: 1.5
    footnote:
      fontSize: 12
      color: "$darkGray"
      fontFamily: "QuattrocentoSans"
      lineHeight: 1.3
  tableStyles:
    default:
      fontSize: 14
      fontFamily: "QuattrocentoSans"
      headerFill: "$primary"
      headerColor: "#FFFFFF"
      headerBold: true
      bodyFill: ["#FFFFFF", "$lightGray"]
      bodyColor: "$text"
      border:
        style: solid
        width: 1
        color: "$midGray"
```
