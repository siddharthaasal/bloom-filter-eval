"""
Wrapper classes for different probabilistic filters to provide a unified interface.
"""

from pybloom_live import BloomFilter as PyBloom
from pybloom_live import ScalableBloomFilter as PyScalableBloom

# CountingBloomFilter is not always directly available in pybloom-live or has issues,
# but usually it's there. Let's try importing.
try:
    from pybloom_live import CountingBloomFilter as PyCountingBloom
except ImportError:
    PyCountingBloom = None

import cuckoofilter
import config


class BaseFilter:
    def __init__(self, capacity=config.FILTER_CAPACITY, error_rate=config.ERROR_RATE):
        self.capacity = capacity
        self.error_rate = error_rate
        self.filter = None
        self.name = "Base"

    def add(self, key):
        raise NotImplementedError

    def __contains__(self, key):
        raise NotImplementedError

    def size_bits(self):
        """Returns approximate size in bits."""
        return 0


class StandardBloomFilter(BaseFilter):
    def __init__(self, capacity=config.FILTER_CAPACITY, error_rate=config.ERROR_RATE):
        super().__init__(capacity, error_rate)
        self.name = "BloomFilter"
        self.filter = PyBloom(capacity=capacity, error_rate=error_rate)

    def add(self, key):
        self.filter.add(key)

    def __contains__(self, key):
        return key in self.filter

    def size_bits(self):
        return self.filter.num_bits


class ScalableBloomFilter(BaseFilter):
    def __init__(self, capacity=config.FILTER_CAPACITY, error_rate=config.ERROR_RATE):
        # Scalable Bloom Filter starts small and grows.
        super().__init__(capacity, error_rate)
        self.name = "ScalableBloom"
        # mode=SmallSetGrowth is default, ratio=2
        # map 'capacity' to 'initial_capacity' for PyScalableBloom
        self.filter = PyScalableBloom(initial_capacity=capacity, error_rate=error_rate)

    def add(self, key):
        self.filter.add(key)

    def __contains__(self, key):
        return key in self.filter

    def size_bits(self):
        # Sum of bits in all sub-filters
        return sum(f.num_bits for f in self.filter.filters)


class CountingBloomWrapper(BaseFilter):
    def __init__(self, capacity=config.FILTER_CAPACITY, error_rate=config.ERROR_RATE):
        super().__init__(capacity, error_rate)
        self.name = "CountingBloom"
        if PyCountingBloom:
            self.filter = PyCountingBloom(capacity=capacity, error_rate=error_rate)
        else:
            raise ImportError("CountingBloomFilter not found in pybloom_live")

    def add(self, key):
        self.filter.add(key)

    def __contains__(self, key):
        return key in self.filter

    def size_bits(self):
        return self.filter.num_bits


class CuckooFilterWrapper(BaseFilter):
    def __init__(self, capacity=config.FILTER_CAPACITY, error_rate=config.ERROR_RATE):
        super().__init__(capacity, error_rate)
        self.name = "CuckooFilter"
        # cuckoofilter library usually takes capacity.
        # It doesn't support error_rate directly in constructor typicaly.
        # We'll just pass capacity.
        try:
            self.filter = cuckoofilter.CuckooFilter(capacity=capacity)
        except TypeError:
            # Fallback if it takes positional
            self.filter = cuckoofilter.CuckooFilter(capacity)

    def add(self, key):
        self.filter.insert(key)

    def __contains__(self, key):
        return self.filter.contains(key)

    def size_bits(self):
        # This is strictly library dependent.
        # If not available, we return 0 or estimated.
        # For now, let's try to access size in bytes * 8 if available.
        try:
            return self.filter.size * 8
        except:
            # Basic theoretical Cuckoo estimate: ~8-12 bits per item?
            # We can't know for sure without internal access.
            return 0
