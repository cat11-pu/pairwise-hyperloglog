"""hyperloglog.core 的验收用例：桶值拆分、两端修正、合并与误差界。

用例只断言期望的计数结果与不变量：摘要到（桶号，桶值）的拆分口径、空桶统计、
稀疏区间与中等规模区间的估计、估计逼近摘要上界时的修正取值、合并后的覆盖关系、
重复插入的幂等性，以及小基数精确区间的口径。
"""

import math
import unittest

from hyperloglog.core import HyperLogLog
from hyperloglog.core import estimate_from_registers
from hyperloglog.core import harmonic_estimate
from hyperloglog.core import large_range_corrected
from hyperloglog.core import leading_zero_rank
from hyperloglog.core import linear_count


MASK64 = (1 << 64) - 1
MIX_ONE = 0xFF51AFD7ED558CCD
MIX_TWO = 0xC4CEB9FE1A85EC53


def key_hash(key):
    """测试用散列：把元素名整体混合成 64 位摘要，落点稳定、可复算。"""
    digest = 0xCBF29CE484222325
    for byte in key.encode("utf-8"):
        digest = ((digest ^ byte) * 0x100000001B3) & MASK64
    digest = (digest ^ (digest >> 33)) * MIX_ONE & MASK64
    digest = (digest ^ (digest >> 33)) * MIX_TWO & MASK64
    return (digest ^ (digest >> 33)) & MASK64


def names(prefix, count):
    """形如 u-0001 的一批元素名。"""
    return ["%s-%04d" % (prefix, index) for index in range(1, count + 1)]


class RankTests(unittest.TestCase):

    def test_leading_zero_rank_counts_the_free_bits(self):
        self.assertEqual(leading_zero_rank(0, 8), 9)
        self.assertEqual(leading_zero_rank(1, 8), 8)
        self.assertEqual(leading_zero_rank(2, 8), 7)
        self.assertEqual(leading_zero_rank(128, 8), 1)
        self.assertEqual(leading_zero_rank(64, 8), 2)
        self.assertEqual(leading_zero_rank(1, 1), 1)
        self.assertEqual(leading_zero_rank(0, 1), 2)
        self.assertRaises(ValueError, leading_zero_rank, -1, 8)
        self.assertRaises(ValueError, leading_zero_rank, 3, 0)
        self.assertRaises(TypeError, leading_zero_rank, 1.5, 8)

    def test_digest_splits_into_bucket_and_rank(self):
        sketch = HyperLogLog(precision=4, hash_fn=key_hash)
        self.assertEqual(sketch.buckets, 16)
        self.assertEqual(sketch.max_rank, 61)
        self.assertEqual(sketch.position_of(0x10), (0, 60))
        self.assertEqual(sketch.position_of(0x4000000000000001), (1, 2))
        self.assertEqual(sketch.position_of(0x800000000000000F), (15, 1))
        self.assertEqual(sketch.position_of(MASK64), (15, 1))
        sketch.add_digest(0x10)
        sketch.add_digest(0x800000000000000F)
        self.assertEqual(sketch.registers()[0], 60)
        self.assertEqual(sketch.registers()[15], 1)
        self.assertEqual(sum(sketch.registers()), 61)
        self.assertFalse(sketch.add_digest(0x10))
        self.assertLessEqual(max(sketch.registers()), sketch.max_rank)


class SmallRangeTests(unittest.TestCase):

    def test_linear_counting_uses_natural_logarithms(self):
        self.assertAlmostEqual(linear_count(16, 16), 0.0, places=12)
        self.assertAlmostEqual(linear_count(1024, 512),
                               1024.0 * math.log(2.0), places=9)
        self.assertAlmostEqual(linear_count(256, 1),
                               256.0 * math.log(256.0), places=9)
        self.assertAlmostEqual(linear_count(4, 2), 4.0 * math.log(2.0), places=9)
        self.assertRaises(ValueError, linear_count, 16, 0)
        self.assertRaises(TypeError, linear_count, 16, 1.5)

    def test_empty_bucket_count_follows_the_registers(self):
        sketch = HyperLogLog(precision=8, hash_fn=key_hash, exact_limit=0)
        self.assertEqual(sketch.zero_buckets(), 256)
        sketch.add_digest(3)
        self.assertEqual(sketch.zero_buckets(), 255)
        sketch.add_digest(7)
        self.assertEqual(sketch.zero_buckets(), 254)
        self.assertEqual(sketch.registers().count(0), 254)
        untouched = HyperLogLog.from_registers([0] * 256, precision=8)
        self.assertEqual(untouched.zero_buckets(), 256)
        once = HyperLogLog.from_registers([1] * 256, precision=8)
        self.assertEqual(once.zero_buckets(), 0)
        mixed = HyperLogLog.from_registers([0] * 128 + [2] * 128, precision=8)
        self.assertEqual(mixed.zero_buckets(), 128)

    def test_small_cardinality_comes_from_the_empty_buckets(self):
        sketch = HyperLogLog(precision=12, hash_fn=key_hash, exact_limit=0)
        self.assertEqual(sketch.zero_buckets(), sketch.buckets)
        self.assertEqual(sketch.estimate(), 0.0)
        tiny = HyperLogLog(precision=8, hash_fn=key_hash, exact_limit=0)
        for key in names("u", 3):
            tiny.add(key)
        self.assertFalse(tiny.is_exact())
        self.assertGreater(tiny.estimate(), 3.0)
        self.assertLess(tiny.estimate(), 3.1)
        for key in names("u", 300):
            sketch.add(key)
        occupied = sum(1 for value in sketch.registers() if value > 0)
        self.assertEqual(sketch.zero_buckets(), sketch.buckets - occupied)
        self.assertLess(sketch.raw_estimate(), 2.5 * sketch.buckets)
        self.assertGreater(sketch.estimate(), 285.0)
        self.assertLess(sketch.estimate(), 315.0)
        exact = HyperLogLog(precision=8, hash_fn=key_hash, exact_limit=32)
        self.assertTrue(exact.is_exact())
        for key in names("v", 20):
            exact.add(key)
        self.assertEqual(exact.estimate(), 20.0)
        for key in reversed(names("v", 20)):
            exact.add(key)
        self.assertEqual(exact.estimate(), 20.0)
        self.assertTrue(exact.is_exact())
        for key in names("v", 40):
            exact.add(key)
        self.assertFalse(exact.is_exact())
        self.assertNotEqual(exact.estimate(), 20.0)


class ErrorBoundTests(unittest.TestCase):

    def test_harmonic_estimate_weighs_buckets_by_powers_of_two(self):
        self.assertAlmostEqual(harmonic_estimate((0,) * 16),
                               0.673 * 256.0 / 16.0, places=9)
        self.assertAlmostEqual(harmonic_estimate((0,) * 12 + (1,) * 4),
                               0.673 * 256.0 / 14.0, places=9)
        self.assertAlmostEqual(harmonic_estimate((2,) * 16),
                               0.673 * 256.0 / 4.0, places=9)
        self.assertAlmostEqual(harmonic_estimate((0,) * 8 + (1,) * 4
                                                 + (2,) * 4),
                               0.673 * 256.0 / 11.0, places=9)
        self.assertRaises(ValueError, harmonic_estimate, ())

    def test_relative_error_is_the_theoretical_bound(self):
        for precision in (4, 8, 12):
            sketch = HyperLogLog(precision=precision, hash_fn=key_hash)
            self.assertAlmostEqual(
                sketch.relative_error(),
                1.04 / math.sqrt(float(1 << precision)), places=12)

    def test_medium_cardinality_stays_inside_the_error_bound(self):
        sketch = HyperLogLog(precision=8, hash_fn=key_hash, exact_limit=0)
        count = 1200
        for key in names("u", count):
            sketch.add(key)
        self.assertGreater(sketch.raw_estimate(), 2.5 * sketch.buckets)
        bound = 3.0 * sketch.relative_error() * count
        self.assertAlmostEqual(sketch.raw_estimate(), float(count), delta=bound)
        self.assertAlmostEqual(sketch.estimate(), float(count), delta=bound)
        once = sketch.registers()
        first = sketch.estimate()
        zeros = sketch.zero_buckets()
        for key in names("u", count):
            self.assertFalse(sketch.add(key))
        self.assertEqual(sketch.registers(), once)
        self.assertEqual(sketch.estimate(), first)
        self.assertEqual(sketch.zero_buckets(), zeros)


class LargeRangeTests(unittest.TestCase):

    def test_large_range_correction_keeps_the_estimate_under_the_ceiling(self):
        ceiling = float(1 << 16)
        registers = (8,) * 16
        raw = harmonic_estimate(registers)
        self.assertGreater(raw, ceiling / 30.0)
        corrected = large_range_corrected(raw, 16)
        self.assertGreater(corrected, raw)
        self.assertLess(corrected, ceiling)
        self.assertAlmostEqual(1.0 - math.exp(-corrected / ceiling),
                               raw / ceiling, places=12)
        self.assertAlmostEqual(estimate_from_registers(registers, 16),
                               corrected, places=9)
        sketch = HyperLogLog.from_registers(registers, precision=4,
                                            hash_bits=16)
        self.assertAlmostEqual(sketch.estimate(), corrected, places=9)


class MergeTests(unittest.TestCase):

    def test_merged_sketch_keeps_the_largest_bucket_and_the_union(self):
        left = HyperLogLog(precision=8, hash_fn=key_hash, exact_limit=64)
        right = HyperLogLog(precision=8, hash_fn=key_hash, exact_limit=64)
        for key in names("u", 30):
            left.add(key)
        for key in names("v", 30):
            right.add(key)
        self.assertEqual(left.estimate(), 30.0)
        self.assertEqual(right.estimate(), 30.0)
        both = left.merged(right)
        for index, value in enumerate(both.registers()):
            self.assertEqual(value,
                             max(left.registers()[index],
                                 right.registers()[index]))
        self.assertEqual(both.estimate(), 60.0)
        self.assertGreaterEqual(both.estimate(), left.estimate())
        self.assertGreaterEqual(both.estimate(), right.estimate())

        heavy_left = HyperLogLog(precision=8, hash_fn=key_hash, exact_limit=0)
        heavy_right = HyperLogLog(precision=8, hash_fn=key_hash, exact_limit=0)
        for key in names("w", 800):
            heavy_left.add(key)
        for key in names("x", 800):
            heavy_right.add(key)
        heavy = heavy_left.merged(heavy_right)
        self.assertGreaterEqual(
            heavy.estimate(),
            max(heavy_left.estimate(), heavy_right.estimate()))


if __name__ == "__main__":
    unittest.main()
