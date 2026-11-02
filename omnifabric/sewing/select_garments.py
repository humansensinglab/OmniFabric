"""
Choose which of the garments predicted by ChatGarment to texture.

ChatGarment predicts every garment it sees in the photo (e.g. a T-shirt and jeans for a full-body photo), while the user
is often interested in only one of them. This script lists the predicted garments and prints the specification files
of the selected ones (one per line, for scripts/run_pipeline.sh):

  - --garments "lower" (or "upper lower", "1 2", "all") selects non-interactively;
  - otherwise, when run from a terminal and more than one garment was predicted, the user is asked;
  - otherwise (batch jobs, single garment) all garments are selected.

The list and the prompt go to stderr, so stdout only carries the selected paths.
"""
import argparse
import json
import os
import re
import sys


def log(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


def garment_names(spec_paths):
    """Garment names from ChatGarment's text description, e.g. {'upper': 'T-shirt', 'lower': 'jeans'}."""
    names = {}
    for spec in spec_paths:
        output_txt = os.path.join(os.path.dirname(os.path.dirname(spec)), "output.txt")
        if os.path.isfile(output_txt):
            for kind, name in re.findall(r"'(\w+)body_garment': \{'garment_name': '([^']*)'", open(output_txt).read()):
                names["wholebody" if kind == "whole" else kind] = name
    return names


def parse_selection(text, kinds):
    """'lower', 'upper lower', '1 2', '1,2' or 'all'/'' -> list of indices into kinds (None if invalid)."""
    tokens = text.replace(",", " ").lower().split()
    if not tokens or tokens == ["all"] or tokens == ["a"]:
        return list(range(len(kinds)))
    selected = []
    for token in tokens:
        if token.isdigit() and 1 <= int(token) <= len(kinds):
            index = int(token) - 1
        elif token in kinds:
            index = kinds.index(token)
        else:
            return None
        if index not in selected:
            selected.append(index)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("all_paths_json", help="ChatGarment's vis_new/all_json_spec_files.json")
    parser.add_argument("--garments", default="", help='e.g. "lower", "upper lower", "1 2" or "all" (default: ask)')
    args = parser.parse_args()

    spec_paths = json.load(open(args.all_paths_json))
    kinds = [os.path.basename(p).replace("_specification.json", "").replace("valid_garment_", "") for p in spec_paths]
    names = garment_names(spec_paths)

    log("ChatGarment predicted:")
    for i, (kind, spec) in enumerate(zip(kinds, spec_paths)):
        render = os.path.join(os.path.dirname(spec), f"valid_garment_{kind}", f"valid_garment_{kind}_render_front.png")
        log(f"  [{i + 1}] {kind:<9} {names.get(kind, '')}")
        if os.path.isfile(render):
            log(f"      preview: {render}")

    if args.garments:
        selected = parse_selection(args.garments, kinds)
        if selected is None:
            sys.exit(f"Invalid garment selection {args.garments!r}; choose from {kinds} (or their numbers).")
    elif len(kinds) > 1 and sys.stdin.isatty():
        while True:
            log(f"Which garments to texture? e.g. '{kinds[-1]}' or '2', several separated by spaces [all]: ", end="")
            selected = parse_selection(sys.stdin.readline(), kinds)
            if selected is not None:
                break
            log(f"  Please choose from {kinds} or 1-{len(kinds)}.")
    else:
        selected = list(range(len(kinds)))
        if len(kinds) > 1:
            log('Texturing all of them (set GARMENTS, e.g. GARMENTS="lower", to choose).')

    log("Selected: " + " ".join(kinds[i] for i in selected))
    print("\n".join(spec_paths[i] for i in selected))


if __name__ == "__main__":
    main()
