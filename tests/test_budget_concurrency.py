"""The request and cost caps must hold when callers run concurrently.

These are invariant tests, not race detectors: CPython's GIL makes the
check-then-increment in BudgetTracker effectively atomic at this granularity,
and the caps hold here even with the lock removed. They exist to pin the
behaviour the parallel runner depends on, so a future refactor that widens the
critical section - or a free-threaded interpreter that removes the GIL - fails
here rather than silently overspending.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from canyonbench.eval.budget import BudgetTracker
from canyonbench.exceptions import BudgetExceededError
from canyonbench.schemas import BudgetConfig


def _config(*, max_requests: int = 100, max_cost_usd: float = 1.0) -> BudgetConfig:
    return BudgetConfig(
        max_requests=max_requests,
        max_cost_usd=max_cost_usd,
        input_per_million_usd=1.0,
        output_per_million_usd=1.0,
    )


def test_request_cap_is_exact_under_concurrency() -> None:
    cap = 200
    tracker = BudgetTracker(config=_config(max_requests=cap))
    granted = 0

    def attempt() -> bool:
        try:
            tracker.reserve_request()
        except BudgetExceededError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=16) as pool:
        granted = sum(pool.map(lambda _: attempt(), range(cap * 3)))

    # Exactly `cap` reservations may succeed: no more, and no fewer.
    assert granted == cap
    assert tracker.requests == cap


def test_cost_cap_is_never_exceeded_under_concurrency() -> None:
    tracker = BudgetTracker(config=_config(max_cost_usd=1.0))
    per_call = 0.01

    def attempt() -> float:
        try:
            return tracker.record_cost(per_call)
        except BudgetExceededError:
            return 0.0

    with ThreadPoolExecutor(max_workers=16) as pool:
        recorded = sum(pool.map(lambda _: attempt(), range(500)))

    assert tracker.cost_usd <= 1.0 + 1e-9
    assert recorded == pytest.approx(tracker.cost_usd)


def test_increments_are_not_lost_under_concurrency() -> None:
    # A lost += would leave requests below the number of successful calls.
    tracker = BudgetTracker(config=_config(max_requests=10_000))
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda _: tracker.reserve_request(), range(5_000)))
    assert tracker.requests == 5_000


def test_serial_behaviour_is_unchanged() -> None:
    tracker = BudgetTracker(config=_config(max_requests=2, max_cost_usd=0.05))
    tracker.reserve_request()
    tracker.reserve_request()
    with pytest.raises(BudgetExceededError, match="Request cap reached"):
        tracker.reserve_request()

    assert tracker.record_cost(0.04) == pytest.approx(0.04)
    with pytest.raises(BudgetExceededError, match="Cost cap would be exceeded"):
        tracker.record_cost(0.02)


def test_record_tokens_prices_and_charges_once() -> None:
    tracker = BudgetTracker(config=_config(max_cost_usd=10.0))
    charged = tracker.record_tokens(1_000_000, 0)
    assert charged == pytest.approx(1.0)
    assert tracker.cost_usd == pytest.approx(1.0)
