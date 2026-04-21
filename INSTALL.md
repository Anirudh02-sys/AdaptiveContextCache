INSTALLATION / BUILD INSTRUCTIONS 
=================================

This project is a Python package. The instructions below install
all dependencies and install the package from source so you can run the server
and scripts.


0) Supported platforms
----------------------

- Linux is the primary target (these steps assume Ubuntu/Debian).
- Python version required: >= 3.8.1 
- Slight Modifications may be needed based on Python version


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

Optional: install PyTorch only if you plan to use the input summarizer or other
torch-backed embedding/model adapters. On WSL or CPU-only machines, prefer the
CPU wheels unless you have a matching CUDA runtime configured:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

Sanity check import:

```bash
python -c "import gptcache; print('gptcache version:', getattr(gptcache, '__version__', 'unknown'))"
```


5) Set up environment variables
-------------------------------

Create a `.env` file in the repo root with your API credentials:

```bash
VOCAREUM_API_KEY="your-api-key-here"
OPENAI_API_BASE="https://genai.vocareum.com/v1"
```

Or for standard OpenAI:

```bash
OPENAI_API_KEY="sk-..."
OPENAI_API_BASE="https://api.openai.com/v1"
```

Before starting the server, load these into your shell:

```bash
set -a; source .env; set +a
```

**Note:** The config file `config/request_gen.test.json` uses `"api_key_env": "VOCAREUM_API_KEY"`.
If you use a different variable name (e.g. `OPENAI_API_KEY`), update the config to match.


6) Run the API server
---------------------

Start the cache API server with OpenAI-compatible routes enabled:

```bash
python -m gptcache_server.server -s 127.0.0.1 -p 8012 -d /tmp/contextcache_data -o True
```

Health check (in another terminal):

```bash
curl -s http://127.0.0.1:8012/
```


7) Run the request generator (optional)
---------------------------------------

With the server running, open another terminal and run:

```bash
source .venv/bin/activate
set -a; source .env; set +a
python scripts/generate_requests.py --config config/request_gen.test.json
```

This sends test requests using data from `test/` and writes metrics to `data/test_apps/`.


Troubleshooting
---------------

- **"No API key provided"**: Ensure you ran `set -a; source .env; set +a` before starting the server.
- **"Connection refused"**: Check that the server port matches `base_url` in your config JSON.
- **Dry-run mode**: Add `-dr yes` to the server command to skip real LLM calls (useful for testing cache logic only).
