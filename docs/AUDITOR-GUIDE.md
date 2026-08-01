# Auditor guide — Kunsh and Atharva

You are checking that the **generated images match the maps that labelled them**.
You are not labelling anything, not drawing anything, and not judging any model.
There is no model output anywhere in this task.

**Time required: 3–5 hours each.** About 96 views, roughly 2–3 minutes per view.

---

## When this happens

**After** the dataset is generated and **before** any model is run. Nothing that
costs money starts until this is done, and the dataset cannot be frozen without
it. If you find problems, sites get regenerated or dropped — which is exactly
why this comes first.

You will be sent one CSV and one folder of images. You do not need to install
anything, run any code, or have cluster access.

---

## The one rule

**Work alone. Do not compare answers with the other auditor until you have both
submitted.** Two independent judgements are the entire point: the agreement rate
between you is a number that goes in the paper. Comparing notes first destroys
it. If something is genuinely ambiguous, write it in `notes` and move on.

---

## What you receive

```
audit.csv                     <- the file you fill in
audit_assets/
  site_0041__view_c16km_oblique.png    <- one review sheet per row
  site_0057__view_a3km_nadir.png
  ...
```

Each review sheet is a single image with up to 8 labelled panels:

| Panel | What it shows |
|---|---|
| **clean RGB** | The generated view, exactly as a model sees it |
| **target overlay** | The same view with the map's feature painted red |
| **blur: target** | The feature suppressed with heavy blur |
| **blur: distractor** | An unrelated region suppressed the same way |
| **texture: target / distractor** | The same pair, suppressed by texture replacement |
| **frequency: target / distractor** | The same pair, suppressed by frequency removal |

Open the sheet at **100% zoom (actual pixels)**. Do not zoom in past 100% —
whether something is visible *at the image's own resolution* is precisely what
question 2 asks.

---

## Your CSV

Every row has your auditor ID already filled in. You fill four columns with
`yes` or `no`. Nothing else. Leave `notes` empty unless something is wrong.

```csv
site,view,auditor,overlay_aligned,feature_resolvable,obvious_edit_artifact,source_mismatch,notes
site_0041,view_c16km_oblique,AUD-KUNSH,yes,no,no,no,
```

Type exactly `yes` or `no` (lowercase). `y`, `n`, `1`, `0`, `true`, `false` are
also accepted. Anything else — including a blank — is rejected by the loader and
sent back to you.

---

## The four questions

### 1. `overlay_aligned` — does the red overlay sit on the right thing?

Look at the **target overlay** panel.

- **yes** — the red region follows the feature you can see: it runs along the
  road, covers the water, or covers the field.
- **yes** — the feature is too small or faint to see at all, but the red region
  is in a plausible place and does not obviously contradict the image (this is
  common and expected at 16 and 24 km).
- **no** — the red region is clearly on the wrong thing: painted over a hillside
  while the visible road runs elsewhere, or offset from the river by an obvious
  margin.

You are judging *registration*, not whether the feature is visible. A shifted
overlay is the failure. Question 2 handles visibility.

### 2. `feature_resolvable` — can you see the feature at all, at 100% zoom?

Look at the **clean RGB** panel, without the overlay.

- **yes** — you can point at it. You can see the road, the water, or the field
  as a distinct thing in the image.
- **no** — you cannot. It is below the resolution of the image, lost in texture,
  or simply not there.

**This is the most important column in the whole audit.** It is the human half
of the extinction measurement: the benchmark's central claim is that some
features become physically unresolvable at high altitude, and your `no` is what
confirms that independently of the geometry. A confident `no` at 24 km is a
correct and valuable answer, not a failure.

Be strict. "I think I can maybe see a line if I squint" is **no**. Only mark
`yes` if you would bet on it.

Use the overlay panel to know *where* to look, then judge from the clean panel
whether you could have seen it unaided.

### 3. `obvious_edit_artifact` — do the edited panels look pasted?

Look at the **blur / texture / frequency** panels.

- **no** (the good answer) — the edited region looks like a smudged, softened, or
  retextured part of the same photograph. You can tell something was changed; it
  should not look like a foreign object.
- **yes** — there is an obvious rectangle, a hard seam, a colour that belongs to
  a different image, or a repeated tile pattern.

Softness or blurriness is not an artifact. A visible **border** is.

Judge the target and distractor panels the same way. If either looks pasted,
answer `yes`.

### 4. `source_mismatch` — do the image and the map plainly disagree?

- **no** (the good answer) — nothing contradicts the map.
- **yes** — the map clearly asserts something the photo refutes: red painted
  across open desert with no road anywhere; a lake marked where you see dry
  ground; a field marked over bare rock.

Reserve `yes` for cases you would defend out loud. This column triggers a kill
criterion: if more than 10% of audited sites are flagged, the class or region
gets cut. Do not flag "I'm not sure" — write that in `notes` instead.

---

## Quick reference

| Column | Good answer | Flag it when |
|---|---|---|
| `overlay_aligned` | `yes` | The red is on the wrong thing |
| `feature_resolvable` | either is fine | — (this is a measurement, not a check) |
| `obvious_edit_artifact` | `no` | You see a seam, rectangle, or pasted patch |
| `source_mismatch` | `no` | Image plainly refutes the map |

Two of these are checks on the pipeline. `feature_resolvable` is different: it is
**data**, and both answers are equally correct. Do not feel pressure to say
`yes` — the honest answer is the useful one.

---

## When you finish

1. Save the CSV with the **same filename** and the same columns. Do not
   reorder, rename, add, or delete columns.
2. Send it back. Do not send it to the other auditor.
3. After both files arrive, disagreements are resolved by discussion, and the
   agreement rate is reported in the paper.

If you get stuck on a specific view, put a short note in `notes` and answer with
your best judgement. Never leave a cell blank.
