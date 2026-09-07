"""Unit tests for RateLimiter. Time is passed in explicitly via the `now` parameter
(rather than monkeypatching time.monotonic) so these tests are deterministic and don't
depend on real wall-clock time passing during the test run."""

from app.rate_limit import RateLimiter


def test_allows_up_to_max_requests_then_blocks():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    assert limiter.allow("ip-a", now=0.0) is True
    assert limiter.allow("ip-a", now=1.0) is True
    assert limiter.allow("ip-a", now=2.0) is True
    assert limiter.allow("ip-a", now=3.0) is False  # 4th request within the window


def test_keys_are_independent():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    assert limiter.allow("ip-a", now=0.0) is True
    assert limiter.allow("ip-b", now=0.0) is True  # different key, own budget
    assert limiter.allow("ip-a", now=0.1) is False
    assert limiter.allow("ip-b", now=0.1) is False


def test_old_hits_expire_out_of_the_window():
    limiter = RateLimiter(max_requests=2, window_seconds=60)
    assert limiter.allow("ip-a", now=0.0) is True
    assert limiter.allow("ip-a", now=1.0) is True
    assert limiter.allow("ip-a", now=2.0) is False  # window full

    # Once the window has fully rolled past the first two hits, they should
    # no longer count against the limit.
    assert limiter.allow("ip-a", now=61.0) is True
    assert limiter.allow("ip-a", now=62.0) is True
    assert limiter.allow("ip-a", now=62.5) is False
