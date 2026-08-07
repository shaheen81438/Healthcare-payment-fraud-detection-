"""In-process request/prediction counters exposed at /metrics (Prometheus text format).

Deliberately dependency-free: counters + a latency summary are enough for a
single-process service; a multi-worker deployment would swap in prometheus-client
with a shared registry (documented in docs/monitoring.md).
"""

from __future__ import annotations

import threading
from collections import defaultdict


class MetricsStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: dict[tuple[str, str, int], int] = defaultdict(int)
        self.latency_sum_ms: dict[str, float] = defaultdict(float)
        self.latency_count: dict[str, int] = defaultdict(int)
        self.predictions_by_action: dict[str, int] = defaultdict(int)
        self.claims_scored: int = 0

    def observe_request(self, method: str, path: str, status: int, duration_ms: float) -> None:
        with self._lock:
            self.requests[(method, path, status)] += 1
            self.latency_sum_ms[path] += duration_ms
            self.latency_count[path] += 1

    def observe_predictions(self, actions: list[str]) -> None:
        with self._lock:
            self.claims_scored += len(actions)
            for action in actions:
                self.predictions_by_action[action] += 1

    def render(self, model_version: str, model_loaded: bool) -> str:
        with self._lock:
            lines = [
                "# HELP api_requests_total Requests by method, path, status.",
                "# TYPE api_requests_total counter",
            ]
            for (method, path, status), count in sorted(self.requests.items()):
                lines.append(
                    f'api_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}'
                )
            lines += [
                "# HELP api_request_latency_ms_sum Cumulative request latency per path.",
                "# TYPE api_request_latency_ms_sum counter",
            ]
            for path, total in sorted(self.latency_sum_ms.items()):
                lines.append(f'api_request_latency_ms_sum{{path="{path}"}} {total:.2f}')
                lines.append(
                    f'api_request_latency_ms_count{{path="{path}"}} {self.latency_count[path]}'
                )
            lines += [
                "# HELP claims_scored_total Claims scored since process start.",
                "# TYPE claims_scored_total counter",
                f"claims_scored_total {self.claims_scored}",
                "# HELP predictions_by_action_total Predictions by recommended action.",
                "# TYPE predictions_by_action_total counter",
            ]
            for action, count in sorted(self.predictions_by_action.items()):
                lines.append(f'predictions_by_action_total{{action="{action}"}} {count}')
            lines += [
                "# HELP model_loaded 1 if a scoring bundle is attached.",
                "# TYPE model_loaded gauge",
                f"model_loaded {int(model_loaded)}",
                f'model_info{{version="{model_version}"}} 1',
            ]
            return "\n".join(lines) + "\n"


metrics_store = MetricsStore()
