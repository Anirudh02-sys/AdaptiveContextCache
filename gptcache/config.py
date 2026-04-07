from typing import Callable, Dict, List, Optional

from gptcache.utils.error import CacheError


class Config:
    """Pass configuration.

    :param log_time_func: optional, customized log time function
    :type log_time_func: Optional[Callable[[str, float], None]]
    :param similarity_threshold: a threshold ranged from 0 to 1 to filter search results with similarity score higher \
     than the threshold. When it is 0, there is no hits. When it is 1, all search results will be returned as hits.
    :type similarity_threshold: float
    :param prompts: optional, if the request content will remove the prompt string when the request contains the prompt list
    :type prompts: Optional[List[str]]
    :param template: optional, if the request content will remove the template string and only keep the parameter value in the template
    :type template: Optional[str]
    :param auto_flush: it will be automatically flushed every time xx pieces of data are added, default to 20
    :type auto_flush: int
    :param enable_token_counter: enable token counter, default to False
    :type enable_token_counter: bool
    :param input_summary_len: optional, summarize input to specified length.
    :type input_summary_len: Optional[int]
    :param skip_list: for sequence preprocessing, skip those sentences in skip_list.
    :type skip_list: Optional[List[str]]
    :param context_len: optional, the length of context.
    :type context_len: Optional[int]
    :param context_cache_window_len: base default window N (fixed); effective length is ``round(N * overall_factor * app_factor)`` clamped to ``[1, context_cache_window_max]``.
    :type context_cache_window_len: int
    :param context_cache_overall_factor: load-adaptive multiplicative factor on the base default (starts at 1.0; adjusted by load controller).
    :type context_cache_overall_factor: float
    :param context_cache_window_factor_by_app: application_id -> per-app multiplier; missing ids use factor 1.0. Starts empty.
    :type context_cache_window_factor_by_app: Optional[Dict[str, float]]
    :param context_cache_window_max: upper cap on the effective window length.
    :type context_cache_window_max: int
    :param load_adaptive: adaptivecontextcache: load-aware adaptation (minute counters + optional context window scaling).
    :type load_adaptive: bool
    :param load_adaptive_ratio: R > 1: shrink window when load >= R× previous minute; grow when load <= previous/R.
    :type load_adaptive_ratio: float
    :param slo_adaptive: adaptivecontextcache: SLO-aware adaptation (reserved; currently no-op).
    :type slo_adaptive: bool

    Example:
        .. code-block:: python

            from gptcache import Config

            configs = Config(similarity_threshold=0.6)
    """

    def __init__(
            self,
            log_time_func: Optional[Callable[[str, float], None]] = None,
            similarity_threshold: float = 0.75,
            dialuoge_threshold: float = 0.6,
            prompts: Optional[List[str]] = None,
            template: Optional[str] = None,
            auto_flush: int = 20,
            enable_token_counter: bool = True,
            input_summary_len: Optional[int] = None,
            context_len: Optional[int] = None,
            skip_list: List[str] = None,
            data_check: bool = False,
            disable_report: bool = False,
            method: str = "mean", # or "attention",
            model_name: str = "qwq",
            cache_mode: str = "context",  # context | plain | none
            context_cache_window_len: int = 5,
            context_cache_overall_factor: float = 1.0,
            context_cache_window_factor_by_app: Optional[Dict[str, float]] = None,
            context_cache_window_max: int = 32,
            load_adaptive: bool = False,
            load_adaptive_ratio: float = 2.0,
            slo_adaptive: bool = False,
    ):
        if similarity_threshold < 0 or similarity_threshold > 1:
            raise CacheError(
                "Invalid the similarity threshold param, reasonable range: 0-1"
            )
        if cache_mode not in ("context", "plain", "none"):
            raise CacheError(
                "Invalid cache_mode, expected one of: context, plain, none"
            )
        if context_cache_window_len < 1:
            raise CacheError(
                "Invalid context_cache_window_len, expected integer >= 1"
            )
        if context_cache_window_max < 1:
            raise CacheError(
                "Invalid context_cache_window_max, expected integer >= 1"
            )
        if context_cache_window_max < context_cache_window_len:
            raise CacheError(
                "context_cache_window_max must be >= context_cache_window_len"
            )
        if context_cache_overall_factor <= 0:
            raise CacheError(
                "context_cache_overall_factor must be > 0"
            )
        _by_app_f: Dict[str, float] = {}
        if context_cache_window_factor_by_app:
            for key, val in context_cache_window_factor_by_app.items():
                sk = str(key).strip()
                if not sk:
                    raise CacheError(
                        "context_cache_window_factor_by_app keys must be non-empty strings"
                    )
                try:
                    fv = float(val)
                except (TypeError, ValueError) as e:
                    raise CacheError(
                        "context_cache_window_factor_by_app values must be numbers"
                    ) from e
                if fv <= 0:
                    raise CacheError(
                        "context_cache_window_factor_by_app values must be > 0"
                    )
                _by_app_f[sk] = fv
        if load_adaptive_ratio <= 1.0:
            raise CacheError(
                "load_adaptive_ratio must be > 1.0"
            )
        self.log_time_func = log_time_func
        self.similarity_threshold = similarity_threshold
        self.prompts = prompts
        self.template = template
        self.auto_flush = auto_flush
        self.enable_token_counter = enable_token_counter
        self.input_summary_len = input_summary_len
        self.context_len = context_len
        if skip_list is None:
            skip_list = ["system", "assistant"]
        self.skip_list = skip_list
        self.data_check = data_check
        self.disable_report = disable_report
        
        # new_added
        self.dialuoge_threshold = dialuoge_threshold
        self.method = method
        self.model_name = model_name
        self.cache_mode = cache_mode
        self.context_cache_window_len = context_cache_window_len
        self.context_cache_overall_factor = float(context_cache_overall_factor)
        self.context_cache_window_factor_by_app = _by_app_f
        self.context_cache_window_max = context_cache_window_max
        self.load_adaptive = load_adaptive
        self.load_adaptive_ratio = float(load_adaptive_ratio)
        self.slo_adaptive = slo_adaptive
        self.context_emb = None
        self.cur_id = 0
        self.set_use_api = False
        self.context_q = []
        self.context_a = []

    def effective_context_window_len(self, application_id: Optional[str] = None) -> int:
        """Effective window: ``round(N * overall_factor * app_factor)`` in ``[1, max]``."""
        n = int(self.context_cache_window_len)
        overall = float(self.context_cache_overall_factor)
        app_f = 1.0
        if application_id is not None:
            aid = str(application_id).strip()
            if aid and aid in self.context_cache_window_factor_by_app:
                try:
                    app_f = float(self.context_cache_window_factor_by_app[aid])
                except (TypeError, ValueError):
                    app_f = 1.0
        if overall <= 0:
            overall = 1.0
        if app_f <= 0:
            app_f = 1.0
        w = int(round(n * overall * app_f))
        w_max = int(self.context_cache_window_max)
        return max(1, min(w, w_max))