# 🐝 Comb Cell Measure

**A free, open-source tool that measures honeycomb cell size from one photo with a ruler lying on the comb.**

**Use it now, free, in your browser: https://comb-cell-measure.streamlit.app/**

Upload a photo, click two marks on the ruler, click two corners of a box over the cells, and press *Measure*.
It finds the comb pattern, snaps every cell to its walls, and reports:

* the **median cell width** (mm, wall centre to wall centre: the same quantity as a "10 cells = 49 mm" ruler count);
* the **spread** (middle half of the cells) and a histogram, so mixed-size foundationless comb shows up;
* the **whole-comb spacing**, an independent second estimate from the comb's repeat pattern;
* a check of your two ruler marks against the ruler's own millimetre ticks;
* an indicative **size class** (natural / small-cell worker, standard worker, intermediate, drone);
* downloads: a cell-by-cell table (CSV) and the marked-up photo (PNG).

Photos are processed in memory and are not stored.

## How accurate is it?

So far this is a **pilot, not a validation.** On two test photos with a printed ground truth, the median and the
whole-comb spacing were both within 0.06 mm (4.80 mm comb: 4.795 / 4.782 mm; 5.30 mm foundation: 5.242 mm by
spacing). Those were small web images where single cells scatter by about ±0.1 mm. The medians are the numbers to
use. A proper validation on 10–20 full-resolution photos with hand counts is planned. If you can contribute
photos, please get in touch: see *Contributing photos* below.

## Taking a good photo

1. Lay the ruler **flat on the comb**, metric side up, across the cells you want to measure.
2. Shoot **straight on**, phone parallel to the comb, about 15–25 cm away. Each cell should be at least ~50 pixels
   across.
3. Use the **full-size photo**, not a copy sent through a messaging app, which compresses it.
4. Use **even light**, with no harsh shadow across the ruler. Empty or backlit comb is easiest; capped brood works.

## Run it yourself

```bash
pip install -r requirements.txt
streamlit run app/bee_app.py
```

Or use the hosted version: https://comb-cell-measure.streamlit.app/

## Contributing photos

To help validate the tool, send comb photos taken as above, with your own "10 cells = __ mm" count if you have one.
Say whether it's worker or drone comb, and whether it's foundation (which size) or natural / foundationless. Every
contributor gets a free measurement report, and is credited by name or kept anonymous, as they prefer.
Contact: *to be added*.

## How it works

The measuring engine (`hexengine/`) began as a research tool for testing hexagon geometry in nature. Its key steps:

* **Ruler scale:** two clicked marks give pixels per mm; the tick period along the ruler (autocorrelation) is an
  independent check.
* **Comb lattice:** the comb's two repeat vectors come from the image's autocorrelation; the six neighbour spacings
  must agree within 15 %.
* **Sub-pixel wall snapping:** each cell's six sides are found along the wall centrelines and fitted with straight
  lines, and the corners are their intersections, with a bootstrap error. A quality gate rejects blurred, damaged
  or covered cells.
* **Width:** the mean of the three opposite-wall distances.

The full engine (geometry research app, tests, experiments and a write-up of the original hypothesis test) is
described in [ENGINE.md](ENGINE.md), [CLOSING_NOTE.md](CLOSING_NOTE.md) and
[docs/article/article.md](docs/article/article.md).

## Credits

Created by Clare Rudling, with Ben. Engine and app built with Claude (Anthropic); the original engine specification
was drafted with Grok (xAI). Released under the MIT licence (see [LICENSE](LICENSE)).
