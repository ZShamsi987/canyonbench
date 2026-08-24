"""Hard request and cost caps for API-backed evaluation."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from canyonbench.exceptions import BudgetExceededError
from canyonbench.schemas import BudgetConfig


@dataclass
class BudgetTracker:
    """Run-wide caps on paid requests and spend.

    Both counters are guarded by a lock. ``reserve_request`` and ``record_cost``
    are check-then-modify sequences, so without one two callers can each observe
    the cap as not yet reached and both proceed past it, and ``+=`` can lose an
    update outright. That makes the cap advisory rather than hard, on the two
    numbers that bound real spend. The lock is uncontended in serial use.
    """

    config: BudgetConfig
    requests: int = 0
    cost_usd: float = 0.0
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def reserve_request(self) -> None:
        with self._lock:
            if self.requests + 1 > self.config.max_requests:
                raise BudgetExceededError(
                    f"Request cap reached ({self.requests}/{self.config.max_requests})"
                )
            self.requests += 1

    def record_tokens(self, input_tokens: int, output_tokens: int) -> float:
        cost = (
            input_tokens * self.config.input_per_million_usd
            + output_tokens * self.config.output_per_million_usd
        ) / 1_000_000
        return self.record_cost(cost)

    def record_cost(self, cost: float) -> float:
        """Record an already-priced request against the run-wide hard cap."""

        with self._lock:
            if self.cost_usd + cost > self.config.max_cost_usd + 1e-12:
                raise BudgetExceededError(
                    f"Cost cap would be exceeded: ${self.cost_usd + cost:.4f} > "
                    f"${self.config.max_cost_usd:.4f}"
                )
            self.cost_usd += cost
            return cost
