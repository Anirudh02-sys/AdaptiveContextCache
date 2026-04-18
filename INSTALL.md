INSTALLATION / BUILD INSTRUCTIONS 
=================================

This project is a Python package. The instructions below install
all dependencies and install the package from source so you can run the server
and scripts.


0) Supported platforms
----------------------

- Linux is the primary target (these steps assume Ubuntu/Debian).
- Python version required: >= 3.8.1 


1) System prerequisites (Ubuntu/Debian)
---------------------------------------

Install Python, build tooling, and `jq` (used by `scripts/run_experiments.sh` to edit JSON configs):

```bash
sudo apt-get update
sudo apt-get install -y \
  python3 \
  python3-venv \
  python3-pip \
  python3-dev \
  build-essential \
  git \
  curl \
  jq
```

2) Get the source code
----------------------

If you already have the repo, skip this step.

```bash
git clone https://github.com/zilliztech/GPTCache.git ContextCache
cd ContextCache
```

3) Create and activate a virtual environment
--------------------------------------------

Using the system default `python3`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

If you installed a specific Python (example `python3.10`):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```


4) Install base dependencies and install the package (editable)
--------------------------------------------------------------

This installs dependencies from `requirements.txt` and installs the package from
your local checkout so code changes take effect immediately.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Sanity check import:

```bash
python -c "import gptcache; print('gptcache version:', getattr(gptcache, '__version__', 'unknown'))"
```


5) Run the API server to check
--------------------------------------------------

Run the cache API server:

```bash
python -m gptcache_server.server -s 127.0.0.1 -p 8011 -d /tmp/contextcache_data
```

Health check:

```bash
curl -s http://127.0.0.1:8011/
```

6) Download dataset
--------------------------------------------------

Download the dataset from https://drive.google.com/drive/u/1/folders/10Oo-17nqrvMcHVdhUfLaMV_DI5yOkb2a and place it inside the AdaptiveContextCache folder. It should be placed such that the path is {ROOT_DIR}/AdaptiveContextCache/data.

Move the chosen warmup files among those in the data folder to the correct data_dir folder mentioned in the config in {ROOT_DIR}/AdaptiveContextCache/config/request_gen.example.json

For our experiment, we moved {ROOT_DIR}/AdaptiveContextCache/data/mt1010_warmup_50.jsonl to {ROOT_DIR}/AdaptiveContextCache/data/mt10_50_5_apps/ and set the config parameters accordingly.



