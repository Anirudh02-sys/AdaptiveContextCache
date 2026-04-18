import argparse
import json
import logging
import os
import uuid
import zipfile
from threading import Lock
from typing import Any, Dict, Optional

from gptcache import cache, Cache
from gptcache.config import Config
from gptcache.adapter import openai
from gptcache.adapter.api import (
    get,
    put,
    init_similar_cache,
    init_similar_cache_from_config,
)
from gptcache.processor.pre import last_content
from gptcache.utils import import_fastapi, import_pydantic, import_starlette

import_fastapi()
import_pydantic()

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
import uvicorn
from pydantic import BaseModel


app = FastAPI()
openai_cache: Optional[Cache] = None
cache_dir = ""
cache_file_key = ""
server_mode = "contextcache"
dry_run: bool = False

_application_slos: Dict[str, Dict[str, Any]] = {}
_application_slos_lock = Lock()


def on_application_registry_changed() -> None:
    """Recompute per-app context-window deltas from registered SLO targets.

    Low-latency policy:
    - O(n) over registered applications
    - uses only *targets* (latency_p99_ms lower is stricter; accuracy_slo higher is stricter)
    - assigns additive integer offsets (turns) around 0; negative shrinks, positive grows
    """

    def _safe_float(x: Any) -> Optional[float]:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        if v != v:  # NaN
            return None
        return v

    # Snapshot quickly under the lock.
    with _application_slos_lock:
        items = list(_application_slos.items())

    # If slo_adaptive is off, keep per-app offsets neutral (0).
    # This makes app-level tuning opt-in via --slo-adaptive / config.slo_adaptive.
    if not bool(getattr(cache.config, "slo_adaptive", False)):
        zeros = {app_id: 0 for app_id, _ in items}
        cache.config.context_cache_window_delta_by_app = dict(zeros)
        if openai_cache is not None:
            openai_cache.config.context_cache_window_delta_by_app = dict(zeros)
        return

    # If no apps (or only one), keep deltas empty / neutral.
    if len(items) <= 1:
        cache.config.context_cache_window_delta_by_app = {}
        if openai_cache is not None:
            openai_cache.config.context_cache_window_delta_by_app = {}
        return

    # Build per-app directional score from SLO targets.
    # - latency pressure: lower latency_p99_ms target => stricter (use 1/latency)
    # - accuracy pressure: higher accuracy_slo target => stricter
    lat_inv: Dict[str, float] = {}
    acc: Dict[str, float] = {}
    for app_id, rec in items:
        lat = _safe_float((rec or {}).get("latency_p99_ms"))
        a = _safe_float((rec or {}).get("accuracy_slo"))
        if lat is not None and lat > 0:
            lat_inv[app_id] = 1.0 / lat
        if a is not None:
            acc[app_id] = a

    def _minmax_norm(vals: Dict[str, float]) -> Dict[str, float]:
        if not vals:
            return {}
        vmin = min(vals.values())
        vmax = max(vals.values())
        if vmax <= vmin:
            return {k: 0.5 for k in vals}
        span = vmax - vmin
        return {k: (v - vmin) / span for k, v in vals.items()}

    lat_n = _minmax_norm(lat_inv)
    acc_n = _minmax_norm(acc)

    # Weighted directional score: negative = favor latency (shorter window),
    # positive = favor accuracy (longer window).
    # alpha > beta biases toward latency reduction.
    alpha = 0.8
    beta = 0.2

    # Map the normalized score to an integer turn offset. A strict-latency app
    # (lat_n=1, acc_n=0) gets score=-alpha and is mapped to the full available
    # shrink room (down to window=1). A strict-accuracy app (acc_n=1, lat_n=0)
    # gets score=+beta and is mapped to the full available grow room (up to
    # context_cache_window_max). The baseline_center bias (<0) keeps the average
    # window below the base when apps are mixed — an overall latency improvement.
    base_n = int(cache.config.context_cache_window_len)
    w_max = int(cache.config.context_cache_window_max)
    max_neg = max(0, base_n - 1)            # room to shrink (delta down to 1-base_n)
    max_pos = max(0, w_max - base_n)        # room to grow (delta up to w_max-base_n)
    baseline_center = -0.25                  # turns added to every registered app
    deltas: Dict[str, int] = {}
    for app_id, _ in items:
        score = beta * acc_n.get(app_id, 0.0) - alpha * lat_n.get(app_id, 0.0)
        if score >= 0:
            raw = (score / beta) * max_pos if beta > 0 else 0.0
        else:
            raw = (score / alpha) * max_neg if alpha > 0 else 0.0
        d = int(round(raw + baseline_center))
        # clamp to headroom (defense in depth; effective_context_window_len also clamps)
        d = max(-max_neg, min(max_pos, d))
        deltas[app_id] = d

    cache.config.context_cache_window_delta_by_app = dict(deltas)
    if openai_cache is not None:
        openai_cache.config.context_cache_window_delta_by_app = dict(deltas)


class CacheData(BaseModel):
    prompt: str
    answer: Optional[str] = ""
    application_id: Optional[str] = None


class ApplicationSloIn(BaseModel):
    """SLO targets for an application (aligned with request_gen application_slo_expectations)."""

    latency_p99_ms: float
    accuracy_slo: float


class ApplicationSloOut(BaseModel):
    application_id: str


def last_content_query_only(data, **kwargs):
    """Return only the current query for GPTCache mode."""
    content = last_content(data, **kwargs)
    return content[0] if isinstance(content, tuple) else content


@app.get("/")
async def hello():
    return "hello gptcache server"


@app.post("/put")
async def put_cache(cache_data: CacheData) -> str:
    put(cache_data.prompt, cache_data.answer)
    return "successfully update the cache"


@app.post("/get")
async def get_cache(cache_data: CacheData) -> CacheData:
    result = get(cache_data.prompt)
    return CacheData(
        prompt=cache_data.prompt,
        answer=result,
        application_id=cache_data.application_id,
    )


@app.post("/flush")
async def flush_cache() -> str:
    cache.flush()
    return "successfully flush the cache"


@app.post("/v1/applications", response_model=ApplicationSloOut)
async def register_application_slo(body: ApplicationSloIn) -> ApplicationSloOut:
    """Register latency and accuracy SLOs; returns a server-issued application id."""
    if body.latency_p99_ms <= 0:
        raise HTTPException(
            status_code=400,
            detail="latency_p99_ms must be positive",
        )
    if not 0.0 <= body.accuracy_slo <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="accuracy_slo must be between 0.0 and 1.0",
        )
    app_id = str(uuid.uuid4())
    record = {
        "latency_p99_ms": body.latency_p99_ms,
        "accuracy_slo": body.accuracy_slo,
    }
    with _application_slos_lock:
        _application_slos[app_id] = record
    on_application_registry_changed()
    return ApplicationSloOut(application_id=app_id)


@app.delete("/v1/applications/{application_id}")
async def deregister_application(application_id: str) -> dict:
    """Remove a previously registered application (SLO record) by id."""
    aid = (application_id or "").strip()
    if not aid:
        raise HTTPException(status_code=400, detail="application_id is required")
    with _application_slos_lock:
        if aid not in _application_slos:
            raise HTTPException(
                status_code=404, detail="unknown application_id"
            )
        del _application_slos[aid]
    on_application_registry_changed()
    return {"ok": True}


@app.get("/cache_file")
async def get_cache_file(key: str = "") -> FileResponse:
    global cache_dir
    global cache_file_key
    if cache_dir == "":
        raise HTTPException(
            status_code=403,
            detail="the cache_dir was not specified when the service was initialized",
        )
    if cache_file_key == "":
        raise HTTPException(
            status_code=403,
            detail="the cache file can't be downloaded because the cache-file-key was not specified",
        )
    if cache_file_key != key:
        raise HTTPException(status_code=403, detail="the cache file key is wrong")
    zip_filename = cache_dir + ".zip"
    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(cache_dir):
            for file in files:
                zipf.write(os.path.join(root, file))
    return FileResponse(zip_filename)


@app.api_route(
    "/v1/chat/completions",
    methods=["POST", "OPTIONS"],
)
async def chat(request: Request):
    if openai_cache is None:
        raise HTTPException(
            status_code=500,
            detail=f"the gptcache server doesn't open the openai completes proxy",
        )

    import_starlette()
    from starlette.responses import StreamingResponse, JSONResponse
    from starlette.concurrency import run_in_threadpool

    openai_params = await request.json()
    is_stream = openai_params.get("stream", False)
    headers = request.headers
    auth_header = headers.get("authorization", None)
    openai_key = auth_header.split(" ")[1] if auth_header else ""
    cache_skip = openai_params.pop("cache_skip", False)
    if server_mode == "no-cache":
        cache_skip = True
    elif cache_skip is False:
        messages = openai_params.get("messages")
        if "/cache_skip " in messages[0]["content"]:
            cache_skip = True
            content0 = openai_params.get("messages")[0]["content"]
            openai_params.get("messages")[0]["content"] = str(content0).replace("/cache_skip ", "")
        elif "/cache_skip " in messages[-1]["content"]:
            cache_skip = True
            content0 = openai_params.get("messages")[-1]["content"]
            openai_params.get("messages")[-1]["content"] = str(content0).replace("/cache_skip ", "")
        print("cache_skip:", cache_skip)
    print("messages:", openai_params.get("messages"))
    try:
        if is_stream:
            def generate():
                for stream_response in openai.ChatCompletion.create(
                    cache_obj=openai_cache,
                    cache_skip=cache_skip,
                    cache_mode=server_mode,
                    api_key=openai_key,
                    dry_run=dry_run,
                    **openai_params,
                ):
                    if stream_response == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    yield f"data: {json.dumps(stream_response)}\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            openai_response = await run_in_threadpool(
                openai.ChatCompletion.create,
                cache_obj=openai_cache,
                cache_skip=cache_skip,
                cache_mode=server_mode,
                api_key=openai_key,
                dry_run=dry_run,
                **openai_params,
            )
            return JSONResponse(content=openai_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"openai error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--host", default="localhost", help="the hostname to listen on"
    )
    parser.add_argument(
        "-p", "--port", type=int, default=8000, help="the port to listen on"
    )
    parser.add_argument(
        "-d", "--cache-dir", default="gptcache_data", help="the cache data dir"
    )
    parser.add_argument("-k", "--cache-file-key", default="", help="the cache file key")
    parser.add_argument(
        "-f", "--cache-config-file", default=None, help="the cache config file"
    )
    parser.add_argument(
        "-o",
        "--openai",
        type=bool,
        default=False,
        help="whether to open the openai completes proxy",
    )
    parser.add_argument(
        "-of",
        "--openai-cache-config-file",
        default=None,
        help="the cache config file of the openai completes proxy",
    )
    parser.add_argument(
        "-m",
        "--server-mode",
        choices=["no-cache", "gptcache", "contextcache", "adaptivecontextcache"],
        default="contextcache",
        help=(
            "chat proxy cache mode: no-cache, gptcache, contextcache, or "
            "adaptivecontextcache (same as contextcache for now; use --load-adaptive / --slo-adaptive for future behavior)."
        ),
    )
    parser.add_argument(
        "--load-adaptive",
        action="store_true",
        help="enable load-adaptive minute counters and context-window scaling.",
    )
    parser.add_argument(
        "--load-adaptive-ratio",
        type=float,
        default=2.0,
        metavar="R",
        help=(
            "multiplicative load factor R>1 for request counts vs previous minute: "
            "shrink when load >= R×; grow when load <= 1/R× (default: 2.0)."
        ),
    )
    parser.add_argument(
        "--load-adaptive-token-ratio",
        type=float,
        default=4.0,
        metavar="R_TOK",
        help=(
            "stricter factor R_TOK>1 for total input-token counts (minute-over-minute). "
            "Must exceed moderate token noise; default 4.0."
        ),
    )
    parser.add_argument(
        "--load-adaptive-shrink-min-rps",
        type=float,
        default=0.0,
        metavar="RPS",
        help=(
            "absolute request-rate floor (req/s) gating shrink events: a ratio spike "
            "only shrinks the window when current-window rate >= RPS. 0 disables the "
            "gate (ratio check alone decides). Useful to suppress warmup->steady ramps."
        ),
    )
    parser.add_argument(
        "--load-adaptive-grow-max-rps",
        type=float,
        default=0.0,
        metavar="RPS",
        help=(
            "absolute request-rate ceiling (req/s) gating grow events: a ratio drop "
            "only grows the window when current-window rate <= RPS. 0 disables."
        ),
    )
    parser.add_argument(
        "--load-adaptive-force-shrink-rps",
        type=float,
        default=0.0,
        metavar="RPS",
        help=(
            "absolute request-rate threshold (req/s) that forces a shrink each "
            "evaluation regardless of ratio. Use when sustained load exceeds server "
            "capacity and minute-over-minute ratios plateau. 0 disables."
        ),
    )
    parser.add_argument(
        "--load-adaptive-spike-bypass-min-prev-req",
        type=float,
        default=0.0,
        metavar="N",
        help=(
            "when shrink_min_rps would veto a ratio spike, still shrink if prev_req>=N "
            "and curr_req>=R×prev_req. 0 disables (default: 0)."
        ),
    )
    parser.add_argument(
        "--load-adaptive-spike-bypass-min-prev-tok",
        type=float,
        default=0.0,
        metavar="T",
        help=(
            "same for token spike bypass using load-adaptive-token-ratio. 0 disables."
        ),
    )
    parser.add_argument(
        "--slo-adaptive",
        action="store_true",
        help="enable SLO-adaptive aspect (adaptivecontextcache; no-op until implemented).",
    )
    parser.add_argument(
        "-dr",
        "--dry-run",
        choices=["no", "yes"],
        default="no",
        help="whether to run in dry-run mode",
    )
    parser.add_argument(
        "-cw",
        "--context-cache-window-len",
        type=int,
        default=5,
        metavar="N",
        help=(
            "max dialogue embeddings kept for contextcache matching (default: 5). "
            "Applies after YAML init too (overrides config.context_cache_window_len from -f)."
        ),
    )
    args = parser.parse_args()
    if args.load_adaptive_ratio <= 1.0:
        parser.error("--load-adaptive-ratio must be > 1.0")
    if args.load_adaptive_token_ratio <= 1.0:
        parser.error("--load-adaptive-token-ratio must be > 1.0")
    if args.load_adaptive_shrink_min_rps < 0:
        parser.error("--load-adaptive-shrink-min-rps must be >= 0")
    if args.load_adaptive_grow_max_rps < 0:
        parser.error("--load-adaptive-grow-max-rps must be >= 0")
    if args.load_adaptive_force_shrink_rps < 0:
        parser.error("--load-adaptive-force-shrink-rps must be >= 0")
    if args.load_adaptive_spike_bypass_min_prev_req < 0:
        parser.error("--load-adaptive-spike-bypass-min-prev-req must be >= 0")
    if args.load_adaptive_spike_bypass_min_prev_tok < 0:
        parser.error("--load-adaptive-spike-bypass-min-prev-tok must be >= 0")

    # Surface gptcache INFO logs (e.g., load_adaptive shrink/grow events) on the server's
    # stderr alongside uvicorn's own logs so experiments can audit window transitions.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("gptcache").setLevel(logging.INFO)
    global cache_dir
    global cache_file_key
    global server_mode
    global dry_run

    if args.cache_config_file:
        init_conf = init_similar_cache_from_config(config_dir=args.cache_config_file)
        cache_dir = init_conf.get("storage_config", {}).get("data_dir", "")
    else:
        init_similar_cache(
            args.cache_dir,
            config=Config(
                context_cache_window_len=args.context_cache_window_len,
                load_adaptive=args.load_adaptive,
                load_adaptive_ratio=args.load_adaptive_ratio,
                load_adaptive_token_ratio=args.load_adaptive_token_ratio,
                load_adaptive_shrink_min_rps=args.load_adaptive_shrink_min_rps,
                load_adaptive_grow_max_rps=args.load_adaptive_grow_max_rps,
                load_adaptive_force_shrink_rps=args.load_adaptive_force_shrink_rps,
                load_adaptive_shrink_spike_bypass_min_prev_req=(
                    args.load_adaptive_spike_bypass_min_prev_req
                ),
                load_adaptive_shrink_spike_bypass_min_prev_tok=(
                    args.load_adaptive_spike_bypass_min_prev_tok
                ),
                slo_adaptive=args.slo_adaptive,
            ),
        )
        cache_dir = args.cache_dir
    cache.config.context_cache_window_len = args.context_cache_window_len
    cache.config.load_adaptive = args.load_adaptive
    cache.config.load_adaptive_ratio = args.load_adaptive_ratio
    cache.config.load_adaptive_token_ratio = args.load_adaptive_token_ratio
    cache.config.load_adaptive_shrink_min_rps = args.load_adaptive_shrink_min_rps
    cache.config.load_adaptive_grow_max_rps = args.load_adaptive_grow_max_rps
    cache.config.load_adaptive_force_shrink_rps = args.load_adaptive_force_shrink_rps
    cache.config.load_adaptive_shrink_spike_bypass_min_prev_req = (
        args.load_adaptive_spike_bypass_min_prev_req
    )
    cache.config.load_adaptive_shrink_spike_bypass_min_prev_tok = (
        args.load_adaptive_spike_bypass_min_prev_tok
    )
    cache.config.slo_adaptive = args.slo_adaptive
    cache_file_key = args.cache_file_key
    server_mode = args.server_mode
    dry_run = args.dry_run == "yes"

    if args.openai:
        global openai_cache
        openai_cache = Cache()
        if args.openai_cache_config_file:
            init_similar_cache_from_config(
                config_dir=args.openai_cache_config_file,
                cache_obj=openai_cache,
            )
        else:
            pre_func = last_content_query_only if server_mode == "gptcache" else last_content
            openai_cache_dir = os.path.join(args.cache_dir, "openai_server_cache")
            init_similar_cache(
                data_dir=openai_cache_dir,
                pre_func=pre_func,
                cache_obj=openai_cache,
                config=Config(
                    context_cache_window_len=args.context_cache_window_len,
                    load_adaptive=args.load_adaptive,
                    load_adaptive_ratio=args.load_adaptive_ratio,
                    load_adaptive_token_ratio=args.load_adaptive_token_ratio,
                    load_adaptive_shrink_min_rps=args.load_adaptive_shrink_min_rps,
                    load_adaptive_grow_max_rps=args.load_adaptive_grow_max_rps,
                    load_adaptive_force_shrink_rps=args.load_adaptive_force_shrink_rps,
                    load_adaptive_shrink_spike_bypass_min_prev_req=(
                        args.load_adaptive_spike_bypass_min_prev_req
                    ),
                    load_adaptive_shrink_spike_bypass_min_prev_tok=(
                        args.load_adaptive_spike_bypass_min_prev_tok
                    ),
                    slo_adaptive=args.slo_adaptive,
                ),
            )
        openai_cache.config.context_cache_window_len = args.context_cache_window_len
        openai_cache.config.load_adaptive = args.load_adaptive
        openai_cache.config.load_adaptive_ratio = args.load_adaptive_ratio
        openai_cache.config.load_adaptive_token_ratio = args.load_adaptive_token_ratio
        openai_cache.config.load_adaptive_shrink_min_rps = args.load_adaptive_shrink_min_rps
        openai_cache.config.load_adaptive_grow_max_rps = args.load_adaptive_grow_max_rps
        openai_cache.config.load_adaptive_force_shrink_rps = args.load_adaptive_force_shrink_rps
        openai_cache.config.load_adaptive_shrink_spike_bypass_min_prev_req = (
            args.load_adaptive_spike_bypass_min_prev_req
        )
        openai_cache.config.load_adaptive_shrink_spike_bypass_min_prev_tok = (
            args.load_adaptive_spike_bypass_min_prev_tok
        )
        openai_cache.config.slo_adaptive = args.slo_adaptive

        import_starlette()
        from starlette.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
