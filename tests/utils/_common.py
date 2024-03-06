import contextlib
import functools
import random
import re
import warnings

import torch


def set_rng_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)


assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)


def assert_not_equal(actual, expected):
    try:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    except AssertionError:
        pass
    else:
        raise AssertionError(f"actual: {actual} and expected: {expected} are equal")


def cache(fn):
    """Similar to :func:`functools.cache` (Python >= 3.8) or :func:`functools.lru_cache` with infinite cache size,
    but this also caches exceptions.
    """
    sentinel = object()
    out_cache = {}
    exc_tb_cache = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = args + tuple(kwargs.values())

        out = out_cache.get(key, sentinel)
        if out is not sentinel:
            return out

        exc_tb = exc_tb_cache.get(key, sentinel)
        if exc_tb is not sentinel:
            raise exc_tb[0].with_traceback(exc_tb[1])

        try:
            out = fn(*args, **kwargs)
        except Exception as exc:
            # We need to cache the traceback here as well. Otherwise, each re-raise will add the internal pytest
            # traceback frames anew, but they will only be removed once. Thus, the traceback will be ginormous hiding
            # the actual information in the noise. See https://github.com/pytest-dev/pytest/issues/10363 for details.
            exc_tb_cache[key] = exc, exc.__traceback__
            raise exc

        out_cache[key] = out
        return out

    return wrapper


def cpu_and_cuda():
    import pytest  # noqa

    return ("cpu", pytest.param("cuda", marks=pytest.mark.needs_cuda))


def needs_cuda(test_func):
    import pytest  # noqa

    return pytest.mark.needs_cuda(test_func)


@contextlib.contextmanager
def ignore_jit_no_profile_information_warning():
    # Calling a scripted object often triggers a warning like
    # `UserWarning: operator() profile_node %$INT1 : int[] = prim::profile_ivalue($INT2) does not have profile information`
    # with varying `INT1` and `INT2`. Since these are uninteresting for us and only clutter the test summary, we ignore
    # them.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=re.escape("operator() profile_node %"),
            category=UserWarning,
        )
        yield


@contextlib.contextmanager
def freeze_rng_state():
    rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()
    yield
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng_state)
    torch.set_rng_state(rng_state)
