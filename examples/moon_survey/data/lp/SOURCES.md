# Lunar Prospector sources

`uv run src/setup.py data` downloads every file in the table below that is not already in this folder and refuses a
file whose sha256 differs. Each row reads: file, url, size in bytes, sha256, resolution and columns used.

| file | url | size | sha256 | resolution and columns used |
| --- | --- | --- | --- | --- |
| `lpgrs_high1_elem_abundance_2deg.tab` | https://pds-geosciences.wustl.edu/lunar/lp-l-grs-5-elem-abundance-v1/lp_9001/data/lpgrs_high1_elem_abundance_2deg.tab | 9949280 | `e866953499813ec089067f1985dfb92fe7adc8957a2121bb115a3b27ea14e946` | 2 degree equal-area pixels, 11306 rows; W_AL2O3, TIO2, W_FEO (weight fractions), W_TH (ppm) and the error-matrix diagonal E[1,1], E[4,4], E[5,5], E[7,7] |
| `hydrogenlow.txt` | https://pds-geosciences.wustl.edu/missions/lunarp/reduced/hydrogenlow.txt | 622233 | `2087d53086674f213427eb10edf1ffc996d266448037a097a0eab6559c4b630e` | 2 degree (60 km) equal-area pixels, 11306 rows; hydrogen in ppm by weight, the fifth column |

## How the values are read

- Both files hold 11306 equal-area pixels on the same 2 degree grid, so a region's value is the plain mean of the pixels
  whose centres fall in it, and its error is `sqrt(sum(error_i^2)) / n`.
- The gamma-ray table gives oxides as weight fractions (g/g); FeO, TiO2 and Al2O3 are multiplied by 100 to read in wt%
  and so are their errors. No element-to-oxide conversion is needed: the table holds the oxides themselves. Th is in ppm.
- A pixel's error is the square root of the matching diagonal term of the label's error matrix (the label says the square
  root of the diagonal is the propagated uncertainty); the matrix indexes 0 MgO, 1 Al2O3, 2 SiO2, 3 CaO, 4 TiO2, 5 FeO,
  6 K, 7 Th, 8 U.
- The hydrogen product gives no per-pixel error, so its `error` is 0.0.
- The 5 degree and 20 degree tables of the same data set hold the same elements at coarser resolution; the 2 degree table
  is the finest equal-area table that holds all four. The half degree hydrogen product is on a latitude and longitude
  grid, not equal-area pixels, so the 2 degree one is used.
- A region with no pixel of a quantity reads `None` for it.

## Citations

- Prettyman, T. H. (2012). Lunar Prospector Gamma Ray Spectrometer derived elemental abundances, data set
  LP-L-GRS-5-ELEM-ABUNDANCE-V1.0. NASA Planetary Data System, Geosciences Node.
  https://pds-geosciences.wustl.edu/lunar/lp-l-grs-5-elem-abundance-v1/
- Prettyman, T. H., J. J. Hagerty, R. C. Elphic, W. C. Feldman, D. J. Lawrence, G. W. McKinney, and D. T. Vaniman
  (2006). Elemental composition of the lunar surface: Analysis of gamma ray spectroscopy data from Lunar Prospector.
  Journal of Geophysical Research, 111, E12007.
- Feldman, W. C., S. Maurice, D. J. Lawrence, R. C. Little, S. L. Lawson, O. Gasnault, R. C. Wiens, B. L. Barraclough,
  R. C. Elphic, T. H. Prettyman, J. T. Steinberg, and A. B. Binder (2001). Evidence for water ice near the lunar poles.
  Journal of Geophysical Research, 106(E10), 23231-23251. The hydrogen product is the Lunar Prospector Neutron
  Spectrometer "Hydrogen_low" data set of June 2001 (Feldman et al.), published on the Geosciences Node's Lunar
  Prospector special products page: https://pds-geosciences.wustl.edu/missions/lunarp/reduced_special.html
