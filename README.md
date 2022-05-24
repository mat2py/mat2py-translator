# Language translator from MATLAB to Python

This package provides a utility `mh_python` to translate MATLAB into Python format while keeping most of the MATLAB accent.
The generated Python code hope to be directly runnable with the [mat2py](https://mat2py.org) engine.

This `mh_python` utility was initially developed under fork of [MISS_HIT](https://github.com/florianschanda/miss_hit) 
and then exported to this seperated repository for uploading to [PyPi](https://pypi.org/project/mh-python/).

The development may still happen inside the [fork](https://github.com/mat2py/miss_hit). We hope to merge this utility
back to `miss_hit` after it is stable.

## Usage

Try it online [here](https://translate.mat2py.org/).

```python
# must install 
python3 -m pip install -U mh-python

# recommend install for prettify generated code
python3 -m pip install black isort --upgrade

# convert `.m` file to `.py` file
mh_python tests/mat2np/demo_fft.m --format 
```

## Copyright & License

The initial intention of `mh_python` was to be an advanced analysis tool of [MISS_HIT](https://github.com/florianschanda/miss_hit),
so it is also licensed under the same GNU Affero GPL version 3 (or later) as described in LICENSE.AGPL.

## Acknowledgements
- [MISS_HIT](https://github.com/florianschanda/miss_hit)
- [Black](https://github.com/psf/black)
