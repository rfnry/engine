import importlib
import json
import os
import sys


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
    values = {name: moon.value_of(name, value) for name, value in found.items()}
    print(json.dumps({"region": argv[0], "values": values, "source": moon.sources()}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
