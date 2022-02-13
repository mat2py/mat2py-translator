#!/usr/bin/env python

import argparse
import os
import re
import sys
import chardet
import keyword
from pathlib import Path

sys.path.append((Path(".") / "..").absolute().as_posix())

special_builtins = {
    i + ".m"
    for i in (
        "i",
        "j",
        "pi",
        "eps",
        "true",
        "false",
        "inf",
        "Inf",
        "nan",
        "NaN",
    )
}

matlab_keyword = [
    i.strip() + ".m"
    for i in """
    'break'
    'case'
    'catch'
    'classdef'
    'continue'
    'else'
    'elseif'
    'end'
    'for'
    'function'
    'global'
    'if'
    'otherwise'
    'parfor'
    'persistent'
    'return'
    'spmd'
    'switch'
    'try'
    'while'
""".strip().splitlines()
]  # generated with matlab command `iskeyword`


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-a",
        "--annotation",
        default=False,
        action="store_true",
        help="docstring for generate function",
    )
    ap.add_argument("-d", "--max-level", default=1, help="maximal level")
    ap.add_argument("-o", "--output", required=True, help="Output directory")
    ap.add_argument("toolbox", help="The toolbox path to be converted")
    options = ap.parse_args()

    toolbox = Path(options.toolbox)

    if not toolbox.is_dir():
        ap.error(f"toolbox {toolbox} does not exist")

    build_in_funcs = {}

    for path, dirs, files in os.walk(toolbox):
        for f in files:
            if not f.endswith(".m") or f in (
                "Contents.m",
                "debug.m",
                *matlab_keyword,
                *special_builtins,
            ):
                continue

            sub_toolbox = Path(path).relative_to(toolbox)
            level = len(sub_toolbox.parts)
            if level > int(options.max_level):
                continue

            if re.search(r"([+@]|(demos?|private)$)", sub_toolbox.as_posix()):
                continue

            with open(Path(path) / f, "rb") as fd:
                tmp = fd.read()
                tmp = tmp.decode(chardet.detect(tmp)["encoding"])
            name = f[:-2]
            if name in keyword.kwlist:
                name = "_" + name

            body = ""

            if options.annotation and re.search("[Bb]uilt-?in function", tmp):
                body = "\n".join(f"    #.m {i}" for i in tmp.splitlines())

            if body.strip() != "":
                body = body + "\n"

            build_in_funcs.setdefault(sub_toolbox, []).append(
                f"def {name}(*args):\n"
                f"{body}"
                f'    raise NotImplementedError("{name}")\n'
            )

    for k, v in build_in_funcs.items():
        (Path(options.output) / k.parent).mkdir(parents=True, exist_ok=True)
        with open(Path(options.output) / f"{k}.py", "w") as fd:
            fd.write(
                "# type: ignore \n"
                "from ._internal.array import M \n"
                "from ._internal.helper import matlab_function_decorators \n"
                "from ._internal.package_proxy import numpy as np \n"
                ""
            )
            fd.write("\n".join(v))


if __name__ == "__main__":
    main()
