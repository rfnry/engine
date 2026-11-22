import importlib
import json
import os
import sys
from dataclasses import asdict

NODE = "https://pds-geosciences.wustl.edu"
GRS = f"{NODE}/lunar/lp-l-grs-5-elem-abundance-v1/lp_9001/data/lpgrs_high1_elem_abundance_2deg.tab"
HYDROGEN = f"{NODE}/missions/lunarp/reduced/hydrogenlow.txt"
SOURCE = {"FeO": GRS, "TiO2": GRS, "Al2O3": GRS, "Th": GRS, "H": HYDROGEN}


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("measure: usage: measure.py <region>, such as N00-15/3", file=sys.stderr)
        return 2
    if source := os.environ.get("MOON_SURVEY_SRC"):
        sys.path.append(source)
    moon = importlib.import_module("moon")

    try:
        found = moon.measured(argv[0])
    except (moon.RegionError, moon.DataError) as error:
        print(error, file=sys.stderr)
        return 1
    values = {name: None if value is None else asdict(value) for name, value in found.items()}
    print(json.dumps({"region": argv[0], "values": values, "units": moon.UNITS, "source": SOURCE}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
