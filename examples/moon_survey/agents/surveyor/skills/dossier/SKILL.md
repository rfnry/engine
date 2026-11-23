---
name: dossier
description: Measure a region from the Lunar Prospector data, write its dossier and check it
---
A dossier is one region's resources, written as `dossier.json` in your working directory, in the shape
`dossier.schema.json` beside this file gives:

- `region`: the id you claimed, such as `N00-15/3`.
- `terrain`: `mare`, `highland`, `polar` or `mixed`.
- `values`: each of `water_ice` (wt%), `TiO2` (wt%), `FeO` (wt%), `Al2O3` (wt%), `Th` (ppm) and `He3` (ppb), as
  `{"low": …, "high": …, "confidence": "low" | "medium" | "high", "sources": […]}`, or `{"unknown": "<why>"}` when
  nothing you found says. `low` and `high` are the range of the value across the region, what a mission would find in
  it: take `p10` and `p90` of its pixels, widened to `min` and `max` or beyond where the sources show the region holds
  more, never the uncertainty of the mean. `confidence` is how sure you are of that range. A source is the dataset, a
  mission, an instrument, a paper or a page you fetched; a URL starts with `https://`. `water_ice` is the region's
  average in the top metre of regolith, its range likewise across the region; concentrations in cold traps go in
  `notes`.
- `notes`: what else you found, in words.
- `simulated`: `false`.

`python "$RFNRY_TREE/surveyor/skills/dossier/measure.py" <region>` prints the region's FeO, TiO2, Al2O3 and Th from the
Lunar Prospector gamma-ray spectrometer and its hydrogen (H) from the neutron spectrometer, each as the mean, error,
pixel count, spread (`min`, `p10`, `p90`, `max`) and unit of the region's 2 degree pixels, `null` where none falls in
it, with each one's source URL. Those files are already mounted (`$MOON_SURVEY_DATA`) and `measure.py` reads them, so
`fetch` is for what they do not hold. Your FeO, TiO2, Al2O3 and Th are compared with these within their errors. The
hydrogen read as water, H ppm × 9 / 10,000 wt%, is a ceiling for `water_ice`, not its value: some of the hydrogen is
implanted by the solar wind, so your `water_ice` low must not exceed it.

`python "$RFNRY_TREE/surveyor/skills/dossier/check.py" dossier.json` prints each problem of the dossier on a line of
its own and exits 1, or prints nothing and exits 0 when the dossier passes.
