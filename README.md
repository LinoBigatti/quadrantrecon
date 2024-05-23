# QuadrantRecon

## Installation

### Prerequisites
- [Python 3.12 (64 bits)](https://www.python.org/downloads/release/python-3120/)
- [Git](https://git-scm.com/downloads)
- [CUDA Toolkit 12.5](https://developer.nvidia.com/cuda-downloads) (If you want GPU acceleration)

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

Then we need to install pytorch. Make sure to choose the correct command for your setup.
```bash
# Linux (no CUDA support)
pip install -r requirements_torch.txt --index-url https://download.pytorch.org/whl/cpu

# Linux (with CUDA support)
pip install -r requirements_torch.txt

# Windows (no CUDA support)
py -m pip install requirements_torch.txt

# Windows (with CUDA support)
py -m pip install requirements_torch.txt --index-url https://download.pytorch.org/whl/cu121
```

To finish up, we need to install the other python dependencies:
```bash
# Linux
pip install -r requirements.txt

# Windows
py -m pip install -r requirements.txt```

### Running

The program can be invoked by running main.py using python. You can use the --help command to see possible options.

```bash
# Linux
python main.py --help

# Windows
py main.py --help
```

You can run the program in batch mode by using more than 1 filename.

## Building static binaries

You can build the release binaries by invoking pyinstaller:

```bash
# Linux
python -m pyinstaller main.spec

# Windows
py -m pyinstaller main.spec
```
