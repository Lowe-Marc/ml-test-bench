# Installation
```
brew install python@3.10
python3.10 -m venv .venv
source .venv/bin/activate


brew install cmake
brew install apache-arrow
export CMAKE_PREFIX_PATH="$(brew --prefix apache-arrow)/lib/cmake"
pip install pyarrow


pip install -r requirements.txt
pip install -e .
```
