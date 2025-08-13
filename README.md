# Installation
```
brew install python@3.10
python3.10 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

# Usage
```
source .venv/bin/activate

python main.py generate_data
python main.py transform
python main.py train
python main.py inference
python main.py visualize
```
