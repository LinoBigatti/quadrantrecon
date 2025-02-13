# QuadrantRecon
QuadrantRecon is an image recognition tool designed to crop quadrants from photos of marine ecosystems, as a preprocessing step for [CoralNET]().
As configured by default, this tool will recognize yellow quadrants from the [MBOM Pole to Pole Project]() on any file you select and crop their inside area.
The detection parameters can be configured to detect other colors, crop differently sized images, or even ignore different parts of the image to detect other quadrant-like shapes.

## Installation

You can install QuadrantRecon via pip:

```bash
# Linux
pip install quadrantrecon

# Windows
py -m pip install quadrantrecon
```

Afterwards, we need to download the model weights. You can find the default one [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).
You can change the file name to `sam_vit_h.pth` and place it in the folder where the program is executed to load it automatically, or you can provide your own file with the `--model-path` and `--model-type` flags:

```bash
# Linux
python -m quadrantrecon --model-path sam_vit_h_4b8939.pth --model-type vit_h

# Windows
py -m quadrantrecon --model-path sam_vit_h_4b8939.pth --model-type vit_h
```

## Usage
The program can be invoked by using the [-m flag](https://docs.python.org/3/using/cmdline.html#cmdoption-m) as follows. You can use the --help command to see possible options.

```bash
# Linux
python -m quadrantrecon --help

# Windows
py -m quadrantrecon --help
```

You can run the program in batch mode by using more than 1 filename or specifying an input folder.

## Development setup

### Prerequisites
- [Python 3.12 (64 bits)](https://www.python.org/downloads/release/python-3120/)
- [Git](https://git-scm.com/downloads)
- [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-downloads) (If you want GPU acceleration)

### Installing dependencies
First, we need to install the required submodules. We can do this using git:
```bash
git submodule update --init --recursive
```
If you are using another git client, look up how to initialize submodules with it.

Afterwards, we need to download the model weights. You can find the default one [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).
Change the file name to sam_vit_h.pth and place it in this folder.

Then, we need to install the python dependencies. A virtual env is recommended, but particularly on windows, it might not work correctly, so feel free to skip this step.
You can create one to install the program like this:
```bash
# Linux
python3.12 -m venv venv/
source venv/bin/activate

# Windows
py -m venv venv/
source venv/Scripts/Activate.ps1
```

To finish up, we need to install the python dependencies and the package itself. Make sure to choose the correct command for your setup.
```bash
# Linux (no CUDA support)
pip install . --index-url https://download.pytorch.org/whl/cpu

# Linux (with CUDA support)
pip install .

# Windows (no CUDA support)
py -m pip install .

# Windows (with CUDA support)
py -m pip install . --index-url https://download.pytorch.org/whl/cu121
```
