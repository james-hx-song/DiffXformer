# Differential Transformer

First, install requirements and create virtual env:
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

To prepare dataset, 
```python
python3 -m dataset.shakespeare.prepare
```

Then, train:
```python
python3 train.py
```
