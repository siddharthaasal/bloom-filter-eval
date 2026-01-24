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

    def delete(self, key):
        """Remove a key from the filter. Raises NotImplementedError if not supported."""
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


import hashlib
import math
import struct


class CustomCountingBloomFilter(BaseFilter):
    """
    A custom implementation of Counting Bloom Filter using an array of counters.
    Supports add and delete operations.
    """

    def __init__(self, capacity=config.FILTER_CAPACITY, error_rate=config.ERROR_RATE):
        super().__init__(capacity, error_rate)
        self.name = "CountingBloom"

        # Calculate optimal number of bits (m) and hash functions (k)
        # m = - (n * ln(p)) / (ln(2)^2)
        self.num_bits = int(-(capacity * math.log(error_rate)) / (math.log(2) ** 2))

        # k = (m/n) * ln(2)
        self.num_hashes = int((self.num_bits / capacity) * math.log(2))

        # Array of counters. Using 8-bit integers (max 255) for simplicity and memory savings logic.
        # In a real rigorous impl, we might check for overflow.
        # We simulate this with a list or bytearray. bytearray is more memory efficient in Python.
        self.counters = bytearray(self.num_bits)

        self.count = 0

    def _get_hashes(self, key):
        """Generates k hash indices for a given key string."""
        # Standard approach: Double Hashing to generate k hashes from fewer MD5/SHA calls
        # hash_i = (hash_a + i * hash_b) % m

        # Prepare key
        if isinstance(key, int):
            key_bytes = struct.pack("q", key)  # 64-bit int
        else:
            key_bytes = str(key).encode("utf-8")

        # Use MD5 for speed/distribution balance in python (simpler than mmh3 without external dep)
        # Ideally use mmh3 but we stick to stdlib if possible or minimize external deps
        h = hashlib.md5(key_bytes).digest()

        # Unpack two 64-bit integers from the 16-byte digest
        # (Actually MD5 is 128 bit, so we can get two 64-bit segments)
        upper, lower = struct.unpack("QQ", h)

        indices = []
        for i in range(self.num_hashes):
            idx = (upper + i * lower) % self.num_bits
            indices.append(idx)
        return indices

    def add(self, key):
        for idx in self._get_hashes(key):
            # Increment counter
            if self.counters[idx] < 255:
                self.counters[idx] += 1
        self.count += 1

    def delete(self, key):
        """Deletes a key from the filter. Note: unsafe if false negative logic applies,
        but standard CBF allows it."""
        indices = self._get_hashes(key)

        # First check if it's arguably there (can't delete if any is 0)
        # Standard CBF just decrements
        for idx in indices:
            if self.counters[idx] > 0:
                self.counters[idx] -= 1
        self.count -= 1

    def __contains__(self, key):
        for idx in self._get_hashes(key):
            if self.counters[idx] == 0:
                return False
        return True

    def size_bits(self):
        # 8 bits per counter
        return self.num_bits * 8


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

    def delete(self, key):
        self.filter.delete(key)

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
