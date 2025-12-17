Training helper for Stanford Dogs subset

Quick start (on mac M1/M2/M4):

1. Create a virtual env and install dependencies (use `uv` or pip):

```bash
# using uv (recommended in this repo)
cd /Users/badwolf/projects/HotDogsRu_Maven
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r python-educating/requirements.txt
```

2. List available classes (dry run):

```bash
python python-educating/train.py --data-root python-educating/archive/images/Images --list-classes
```

3. Train on selected breeds (example: beagle only):

```bash
python python-educating/train.py --data-root python-educating/archive/images/Images --selected beagle --epochs 15 --batch_size 32 --use_transfer
```

Notes:
- By default the trained Keras model is saved to `src/main/resources/Keras_Model.h5`.
- For Apple silicon use `tensorflow-macos` + `tensorflow-metal` to accelerate training. If you have access to a server with H100 GPUs, set up the same script there and increase batch size / epochs.
- If you want me to start training here, tell me which classes to train and whether to run locally on your Mac or on the H100 server.
