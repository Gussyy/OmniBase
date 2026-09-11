# Design

The visual system of the service page (`omnibase/ui/index.html`). One file, no build, no framework; tokens are CSS custom properties on `:root`.

## Theme

Light, warm-neutral. The page is an instrument read at a desk beside a terminal; it matches the terminal's register (monospace, rules, no decoration) and the terminal's ground is inverted only for the output band. The one chromatic colour is amber, borrowed from the Neo Mirai reference, and it is spent on state alone.

## Color

| token | value | role |
|---|---|---|
| `--ink` | `oklch(0.18 0.01 70)` | text, rules, the selected step |
| `--ink-2` | `oklch(0.42 0.01 70)` | labels, secondary text (≈8:1 on `--bg`) |
| `--ink-3` | `oklch(0.62 0.01 70)` | hints, ids, placeholders (≈4.6:1 on `--bg`; never for body text) |
| `--bg` | `oklch(0.985 0.003 80)` | page and inputs |
| `--panel` | `oklch(0.955 0.006 80)` | the left column, table headers |
| `--line` / `--line-2` | `oklch(0.86 0.008 80)` / `oklch(0.93 0.006 80)` | borders; hover wash |
| `--dark` / `--dark-ink` | `oklch(0.21 0.012 60)` / `oklch(0.92 0.01 80)` | the output band |
| `--amber` / `--amber-ink` | `oklch(0.76 0.16 68)` / `oklch(0.2 0.02 60)` | run button; dark text on it (≈8:1) |
| `--amber-soft` / `--amber-line` | `oklch(0.94 0.05 78)` / `oklch(0.62 0.15 62)` | the set plan bar, highlighted cells; focus ring, best cell |
| `--bad` / `--bad-soft` | `oklch(0.55 0.19 28)` / `oklch(0.95 0.03 28)` | errors, failed jobs, the robot's own base on the map |

Strategy: restrained. Amber appears on the current plan, the run button, a running job's dot, the focus ring and the best cell — states, never decoration. The map is a warm grey ramp (`oklch(0.95→0.23, chroma 0.01→0.03, hue 70)`), so the amber and red markers are the only saturated marks on it.

## Typography

One family: the system monospace stack (`ui-monospace, "SF Mono", Menlo, Consolas, monospace`), 13px/1.45 for everything, 12px for hints and the output band, 14px bold for the wordmark. Labels are the CLI flags they set (`--hands`, `--pos-tol (m)`), so the page teaches the command line. No display sizes, no uppercase tracking, no eyebrows.

## Layout

Two columns at ≥900px: a 400px panel on the left (plan bar, steps, form, command, jobs) and the result on the right; one column below that, the panel first. Rules are 1px, `--ink` for structural edges (header, panel edge, the step strip), `--line` for everything inside. Forms are a two-column grid, 118px label column, hints under the field. Tables are dense, right-aligned numerals, first column left, headers on the panel tone. Long tables scroll inside `.tw`; the page never scrolls sideways.

## Components

- Step strip: four equal buttons in one bordered strip, number + name + a two-word subtitle; the current step is inverted; steps 2–4 are disabled until a plan exists (`title` says why).
- Plan bar: the artefact the steps share; amber-soft with an amber border when set, `[clear]` as a bracket action.
- Inputs: 1px `--line`, hover `--ink-3`, focus amber border plus a 2px amber-soft ring, `aria-invalid` turns the border and fill to the error pair. Required field marked with an amber asterisk.
- Run: the only filled button. Amber, dark text, 1px amber-line border; disabled goes to `--line` with `cursor: wait`.
- Bracket actions: text links and buttons wrapped in `[` `]` by pseudo-elements; hover to amber-line. Used for result file, use this plan, run again, clear.
- Jobs: a list of rows with a status dot (grey queued, pulsing amber running, ink done, red failed), tool, id, elapsed.
- Output band: the job's stdout in the dark tone, monospace 12px, scrolls, keeps the last line in view.
- Map: a canvas, 520px, warm grey ramp, axes in metres, best cell outlined in amber-line with the word "best", the robot's own base in red with "own"; a text legend beneath it always.

## Motion

180ms `cubic-bezier(0.22, 1, 0.36, 1)` on background, colour, border and shadow changes; the running dot pulses at 1.2s. Nothing moves on load. `prefers-reduced-motion: reduce` removes every transition and animation.

## Voice in the interface

Sentences, lower-case labels, no exclamation marks. Every result carries its unit and, where a number needs a reference to mean anything, the reference rows are printed in the same table in a quieter ink. Where a measurement has a known limit — none of this predicts success — the limit is printed next to the result, not in a footnote.
