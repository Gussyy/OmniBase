# Product

## Register

product

## Platform

web

## Users

Anyone who likes robots and has, or wants to make use of, recordings of a person doing a task with a hand-held gripper: researchers training imitation policies from UMI-style data, engineers deciding where to bolt an arm, people judging what a dataset can teach before spending a GPU on it. They are comfortable with a command line and a dataset on disk, and they would rather see a number with a unit than a score without one. Most arrive with one question — where would the robot have to stand? — and leave with a second: what can this data actually teach?

## Product Purpose

OmniBase turns human demonstrations into something a robot can execute, by choosing where the robot stands, and then tells you what the data can and cannot teach a policy — all by arithmetic, with no simulator, no trained model and no new recordings. Success is a person going from a folder of recordings to a placement, a trainable dataset, and an honest read on augmentation and policy behaviour in an afternoon, and trusting each number because they could see how it was made.

## Positioning

The base pose is a free variable in the data, and everything worth knowing about a robot dataset follows from treating it as one.

## Brand Personality

Precise, plain, honest. The voice of a well-kept lab notebook: units on every number, the command shown next to its result, limits stated in the same breath as claims. It never sells; it measures. Warmth comes from clarity and from the tool doing exactly what it said, not from copy.

## Anti-references

Not the generic AI-generated dashboard: rounded cards in a grid, gradient text, purple accents, hero metrics, an eyebrow label above every section. Not a marketing page for a library — no promises of success rates, no "predicted success" numbers, no claims the measurements do not support. Not a wall of unexplained flags either: the command line is the truth, but the page should not make people memorise it.

## Design Principles

- Show the command. Every action on the page is a command line the user could run themselves, and they can see it before and after it runs.
- One plan, in order. The work has a sequence — place, then report, then evaluate — and the interface keeps the artefact that links them so nothing is copied by hand.
- Numbers carry units and references. A miss is in centimetres; a probe is bracketed by replay and oracle; a yield is against a fixed base. A number nobody can interpret is not shown.
- Say what it does not do. Offline scores do not predict success; the page and the docs say so where the numbers appear.
- Defaults first, everything reachable. The essentials are visible; the rest is a fold away, never removed.

## Accessibility & Inclusion

WCAG 2.1 AA: text contrast at or above 4.5:1 (the muted ink is checked against the panel, not assumed), every control reachable and operable by keyboard with a visible focus state, form errors tied to their field, and all motion suppressed under prefers-reduced-motion. The map's information is also given in text (best base, own base, percentages) so the canvas is never the only carrier.
