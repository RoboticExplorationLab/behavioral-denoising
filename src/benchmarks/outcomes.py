"""Controller failure and input-modification counts for recorded episodes."""

from collections import Counter

import numpy as np


def benchmark_outcomes(settings):
    protocol = settings.get("outcome_protocol")
    if protocol not in {None, "legacy", "benchmark_v1"}:
        raise ValueError(f"Unknown outcome protocol: {protocol}")
    return protocol == "benchmark_v1"


class EpisodeAudit:
    """Count controller decisions and applied-action modifications, not solver iterations."""

    def __init__(self):
        self.solver_status = Counter()
        self.action_sources = Counter()
        self.applied_actions = 0
        self.modified_actions = 0
        self.clipped_actions = 0
        self.noisy_actions = 0
        self.override_actions = 0

    def decision(self, policy):
        diagnostics = policy.diagnostics() if hasattr(policy, "diagnostics") else {}
        status = diagnostics.get("status")
        if status is None:
            status = getattr(policy, "status", None)
        if status is None:
            status = getattr(policy, "last", {}).get("status")
        self.solver_status[str(status) if status is not None else "not_reported"] += 1
        self.action_sources[str(diagnostics.get("action_source") or "policy")] += 1

    def applied(self, proposed, applied, *, clipped=False, noisy=False, override=False):
        self.applied_actions += 1
        self.clipped_actions += int(clipped)
        self.noisy_actions += int(noisy)
        self.override_actions += int(override)
        self.modified_actions += int(override or not np.array_equal(proposed, applied))
        if override:
            self.action_sources["task_override"] += 1

    def fields(self, terminal_reason, error=None):
        failed = terminal_reason in {"history_infeasible", "nonfinite_action", "nonfinite_state"}
        return {
            "terminal_reason": terminal_reason,
            "controller_failure": failed,
            "history_infeasible": terminal_reason == "history_infeasible",
            "controller_error": error,
            "failure_cost": float("inf") if failed else 0.0,
            "solver_status_counts": dict(self.solver_status),
            "action_source_counts": dict(self.action_sources),
            "applied_actions": self.applied_actions,
            "modified_actions": self.modified_actions,
            "clipped_actions": self.clipped_actions,
            "noisy_actions": self.noisy_actions,
            "override_actions": self.override_actions,
        }
