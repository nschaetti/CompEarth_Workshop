# Migrate

Code for Migrate project, demonstration and teaching.

## Installation

Install the package with `pip` once it has been published to PyPI:

```bash
pip install compearth-workshop
```

For local development, install it from the project root instead:

```bash
pip install .
```

The `dispsurf2k25` simulator relies on PyTorch. Install the optional extra if you
need that functionality:

```bash
pip install compearth-workshop[torch]
```

## Usage

```python
import compearth
from compearth.extensions import surfdisp2k25
```
