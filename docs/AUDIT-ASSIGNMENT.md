# Audit assignment — Kunsh and Atharva

Everything you need is in this file. [AUDITOR-GUIDE.md](AUDITOR-GUIDE.md) covers
the same four questions with more discussion; read it once if you want the
reasoning, but you can work entirely from this page.

- [The job](#the-job)
- [Software you need](#software-you-need)
- [Setting up](#setting-up)
- [What a review sheet looks like](#what-a-review-sheet-looks-like)
- [The loop, per view](#the-loop-per-view)
- [The four questions](#the-four-questions)
- [Common mistakes](#common-mistakes)
- [When you finish](#when-you-finish)

---

## The job

You each answer **four yes/no questions about 96 images**. Roughly 2–3 minutes
per image, so **3–5 hours**. You are not labelling anything, not drawing
boundaries, and not judging any model. No model output appears anywhere in this
task.

**Why it blocks everything.** Nothing that costs money runs until both CSVs come
back. If the audit finds problems, sites get regenerated or dropped — and that
cannot happen after models have run against them. It is also the only remaining
step that runs on human time; everything before it is automated and everything
after it is queued behind you.

**The one rule.** Work alone. Do not compare answers with each other until you
have both submitted. The agreement rate between you is a number reported in the
paper, and comparing notes destroys it. If a view is genuinely ambiguous, write
that in `notes` and move on.

---

## Software you need

### For the images: a web browser

Drag the `.png` into **Chrome, Firefox, or Edge**. Browsers open a large image
shrunk to fit and show a magnifier cursor; **click once** and it switches to
100% — one image pixel per screen pixel. Click again to zoom back out.

That toggle is the whole reason to use a browser: you need certainty about
whether you are looking at actual pixels, and the click state tells you.

Alternatives if you prefer:

| Platform | App | How to get 100% |
|---|---|---|
| macOS | Preview | **View → Actual Size** |
| Windows | IrfanView | **View → Original size** |
| Windows | Photos | Zoom slider → set to **100%** |

**Do not zoom past 100%.** Whether something is visible *at the image's own
resolution* is precisely what question 2 asks. Zooming in past 100% invents
detail that the model never had, and it will make you answer `yes` when the
correct answer is `no`.

### For the CSV: Google Sheets or LibreOffice Calc

**Recommended — Google Sheets:**

1. sheets.google.com → **File → Import → Upload** → pick `audit.csv`
2. Import location: **Replace spreadsheet**. Separator type: **Comma**.
3. Work in the sheet.
4. When done: **File → Download → Comma Separated Values (.csv)**

**Also fine — LibreOffice Calc.** On import, set Separator to **Comma** and
Character set to **UTF-8**. Save as CSV and choose **Keep Current Format** when
prompted.

**Avoid Microsoft Excel.** It silently rewrites CSVs — changing encoding, adding
a byte-order mark, and on some regional settings using semicolons instead of
commas. Any of those makes the file fail to load and it comes back to you.

A plain text editor (VS Code, Sublime, Notepad++) is also completely fine if you
are comfortable with it — the file is small and the columns are simple.

---

## Setting up

1. Unzip the packet. You get:

   ```
   audit.csv
   audit_assets/
     site_0041__view_c16km_oblique.png
     site_0057__view_a3km_nadir.png
     ...
   ```

2. Open `audit.csv`. It contains rows for **both** of you — 192 rows total, 96
   each.

3. **Filter to your own rows.** The `auditor` column is either `AUD-KUNSH` or
   `AUD-ATHARVA`. In Sheets: select row 1 → **Data → Create a filter** → click
   the filter icon on `auditor` → tick only your ID.

   You fill in **only your own rows**. Leave the other person's rows untouched —
   do not delete them, and do not look at them once they have answers in.

4. Put the browser and the spreadsheet side by side. You will alternate between
   them about 400 times, so the setup is worth 30 seconds.

The columns you fill are `overlay_aligned`, `feature_resolvable`,
`obvious_edit_artifact`, `source_mismatch`. Type `yes` or `no`, lowercase.
`y`, `n`, `1`, `0`, `true`, `false` also load. **A blank cell is rejected** and
the file comes back to you. `notes` is free text and may be left empty.

---

## What a review sheet looks like

Each PNG is **2048 × 4240 pixels** — eight panels in two columns and four rows,
each panel 1024 × 1024 with its name printed in black above it. At 100% zoom you
will see roughly one panel at a time and scroll down through the rest. That is
expected.

| | Left column | Right column |
|---|---|---|
| **Row 1** | `clean RGB` | `target overlay` |
| **Row 2** | `blur: target` | `blur: distractor` |
| **Row 3** | `texture: target` | `texture: distractor` |
| **Row 4** | `frequency: target` | `frequency: distractor` |

- **`clean RGB`** — the generated view exactly as a model sees it. Nothing marked.
- **`target overlay`** — the same view with the map's feature painted red.
- **Rows 2–4** — six edited versions. Each operator (blur, texture replacement,
  frequency removal) is applied twice: once to the **target** (the mapped
  feature) and once to a **distractor** (an unrelated region of similar size).
  All six are at 100% suppression.

Some sheets have fewer than eight panels. That is normal and not a defect.

The filename tells you the altitude: `a3km`, `b8km`, `c16km`, `d24km`. Expect
features to be obvious at 3 km and often invisible at 24 km — that is the
experiment, not a bug.

---

## The loop, per view

For each of your 96 rows:

1. Read `site` and `view` from the row. The sheet is
   `audit_assets/<site>__<view>.png`.
2. Open it in the browser. **Click once** to get to 100%.
3. Look at **`target overlay`** (top right). Answer `overlay_aligned`.
4. Look at **`clean RGB`** (top left), with the overlay out of your mind.
   Answer `feature_resolvable`.
5. Scroll to rows 2–4. Answer `obvious_edit_artifact`.
6. Answer `source_mismatch` from what you have already seen.
7. Type the four answers. Move to the next row.

Go in the order the file gives you. Do not sort or reorder the rows — the file
must come back with its rows in the same order.

**Before you settle in, do the first five and stop.** Re-read the four questions
below against what you actually saw. It is much cheaper to recalibrate at view 5
than to discover at view 90 that you were reading one question differently.

---

## The four questions

### 1. `overlay_aligned` — is the red on the right thing?

Look at **`target overlay`**. You are judging *registration* — whether the red
sits where the feature is — not whether you can see the feature.

**`yes`** when:
- The red follows the thing you can see: it runs along the road, covers the
  water, covers the field.
- You cannot see the feature at all, but the red is somewhere plausible and
  nothing in the image contradicts it. **This is common and correct at 16 and
  24 km.** Absence of evidence is `yes` here.
- The red is slightly wider or narrower than the feature, or has ragged edges.
  Small mismatches in extent are expected from map generalisation.

**`no`** when:
- The red is plainly on something else — painted over a hillside while the
  visible road runs a few hundred metres away.
- The red is offset from the feature by an obvious margin: you can see the river
  and you can see the red, and they are clearly parallel rather than on top of
  each other.
- The red is rotated or mirrored relative to the visible feature.

The test for `no` is: **you can point at the feature and point at the red, and
they are different places.** If you cannot point at the feature, the answer is
`yes`.

### 2. `feature_resolvable` — can you see it, at 100%, unmarked?

Look at **`clean RGB`**. Use the overlay panel first to learn *where* to look,
then judge from the clean panel whether you could have seen it unaided.

**`yes`** — you can point at it. It is a distinct thing in the image: a line you
can trace, a body of water with an edge, a field with a boundary.

**`no`** — you cannot. It is below the image's resolution, lost in surrounding
texture, or simply not visible.

**Be strict.** "I think I can maybe see a line if I squint" is **`no`**. Only
answer `yes` if you would bet money on pointing at it correctly with the overlay
hidden.

**This is the most important column in the audit, and both answers are equally
correct.** It is not a check on the pipeline — it is *data*. The benchmark's
central claim is that some features become physically unresolvable at altitude,
and your `no` is the human half of that measurement. A confident `no` on a 24 km
road is a valuable result, not a failure to see something.

There is a real psychological pull toward `yes` here, because saying "I can't
see it" feels like admitting you did the task badly. Resist it. The honest
answer is the useful one, and honest `no`s are what the paper needs.

### 3. `obvious_edit_artifact` — do the edits look pasted?

Look at the six panels in rows 2–4. Judge the target and distractor panels the
same way; if **either** looks pasted, answer `yes`.

**`no`** — the good answer. The edited region looks like a smudged, softened, or
retextured part of the same photograph. You can tell something changed, and it
still looks like it belongs to the image.

**`yes`** when you see:
- A hard rectangular boundary, or any straight seam that does not follow
  anything in the scene.
- A colour or brightness inside the region that belongs to a different image.
- A repeated tile pattern — the same patch of ground copied several times.
- A sharp edge where the edit stops, rather than a gradual one.

**Blurriness is not an artifact. A visible border is.** The blur panels are
*supposed* to look blurry — that is the intervention working. What you are
looking for is evidence that a region was pasted in rather than degraded in
place.

### 4. `source_mismatch` — do the image and the map plainly disagree?

**`no`** — the good answer. Nothing in the image contradicts the map.

**`yes`** when the map asserts something the photograph refutes:
- Red painted across open desert with no road anywhere in the frame.
- A lake marked where you can plainly see dry ground.
- A field marked over bare rock or forest.

**Reserve `yes` for cases you would defend out loud.** This column triggers a
kill criterion: if more than 10% of audited sites are flagged, a whole class or
region gets cut from the study. Do not flag uncertainty — write it in `notes`
instead.

Note the difference from question 2: "I cannot see a road" is `feature_resolvable
= no`, **not** `source_mismatch = yes`. Mismatch means the image shows something
that actively contradicts the map, not that it fails to confirm it.

---

## Common mistakes

| Mistake | Why it matters |
|---|---|
| Zooming past 100% to find the feature | Invents detail the model never had; turns a correct `no` into a wrong `yes` on the most important column |
| Answering `feature_resolvable = no` and then `source_mismatch = yes` | Not seeing something is not the same as the image refuting it. Invisible is question 2; contradicted is question 4 |
| Marking `overlay_aligned = no` because the feature is invisible | That is question 2. If you cannot locate the feature, the overlay is not misaligned — it is unverifiable, which is `yes` |
| Marking `obvious_edit_artifact = yes` because a panel is blurry | Blur is the intervention working as designed. Look for seams and borders |
| Reordering or sorting the CSV | The file will not load |
| Editing the other auditor's rows | Destroys the independent-agreement measurement |
| Leaving a cell blank | Rejected by the loader; the file comes back to you |
| Opening the CSV in Excel | Silently rewrites encoding and delimiters |

---

## When you finish

1. Save the CSV under the **same filename** with the **same columns**, in the
   **same row order**. Do not reorder, rename, add, or delete columns.
2. Send it back. **Not to each other.**
3. Once both files arrive, disagreements are resolved by discussion, and the
   agreement rate goes into the paper.

Never leave a cell blank. If one view defeats you, put a short note in `notes`
and answer with your best judgement.

**If the packet itself is broken** — a missing sheet, an image that will not
open, a row whose file does not exist — say so immediately rather than working
around it. That is a defect in the generated packet, and regenerating it is
faster and safer than patching around it by hand.

---

## Timing

The submission deadline is **30 August 2026** (VLM4RWD, NeurIPS 2026). The
packet is generated automatically at the end of the dataset build and sent to
you as soon as it exists.

Turning your CSVs around within a day of receiving them keeps the model runs,
analysis, and writing on schedule.
