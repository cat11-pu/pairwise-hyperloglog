"""基数估计内核：分桶寄存器、前导零计数、调和平均估计与两端修正。

内核只做纯计算：元素由调用方交给 add，散列函数也由调用方注入，因此同一串
输入在任何进程里都会得到完全相同的寄存器与估计值。模块不使用随机数、时钟、
网络或任何 I/O，内存里只有 m 个整数寄存器。

约定：

* 桶数 m 等于 1 左移精度位，精度取 4 到摘要位宽减 2；
* 摘要按低 精度 位取桶号，其余高位是待数前导零的尾段；
* 桶值等于尾段写成 摘要位宽减精度 位后前导零的个数加一，全零尾段取最大值
  摘要位宽减精度加一，同一个桶里的桶值只增不减；
* 去重元素数不超过精确区间上限时，估计直接等于去重后的元素个数；
* 离开精确区间后，桶值稀疏时走线性计数，估计逼近摘要上界时走进大基数修正，
  其余情况走调和平均。

对外接口：

* 纯函数 leading_zero_rank / alpha_for / harmonic_estimate / linear_count /
  large_range_corrected / corrected_estimate / estimate_from_registers；
* 分桶寄存器 HyperLogLog：add / add_digest / position_of / registers /
  zero_buckets / raw_estimate / estimate / relative_error / merged /
  from_registers。
"""

import math

MASK64 = (1 << 64) - 1
FNV_OFFSET_BASIS = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
DEFAULT_PRECISION = 12
DEFAULT_HASH_BITS = 64
DEFAULT_EXACT_LIMIT = 64

__all__ = [
    "HyperLogLog",
    "HyperLogLogError",
    "alpha_for",
    "corrected_estimate",
    "estimate_from_registers",
    "fnv1a64",
    "harmonic_estimate",
    "large_range_corrected",
    "leading_zero_rank",
    "linear_count",
]


class HyperLogLogError(Exception):
    """基数估计内核在参数或操作不合法时抛出的异常。"""


def _require_int(value, label):
    """确认参数是非布尔的整数。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s 必须是整数: %r" % (label, value))
    return value


def _as_bytes(key):
    """字符串或字节串统一成字节串。"""
    if isinstance(key, str):
        return key.encode("utf-8")
    if isinstance(key, (bytes, bytearray)):
        return bytes(key)
    raise TypeError("元素必须是字符串或字节串: %r" % (key,))


def fnv1a64(key):
    """默认散列：FNV-1a 的 64 位摘要，只做整数运算、随时可复算。"""
    digest = FNV_OFFSET_BASIS
    for byte in _as_bytes(key):
        digest ^= byte
        digest = (digest * FNV_PRIME) & MASK64
    return digest


def leading_zero_rank(value, width):
    """把 value 写成 width 位，数出前导零个数再加一；全零取 width 加一。"""
    _require_int(value, "待数前导零的值")
    _require_int(width, "位宽")
    if value < 0:
        raise ValueError("待数前导零的值不能为负: %r" % (value,))
    if width < 1:
        raise ValueError("位宽必须为正: %r" % (width,))
    if value == 0:
        return width + 1
    return width - value.bit_length() + 1


def alpha_for(buckets):
    """估计常量：桶数是 16、32、64 时取离散值，其余按渐近式。"""
    if buckets == 16:
        return 0.673
    if buckets == 32:
        return 0.697
    if buckets == 64:
        return 0.709
    return 0.7213 / (1.0 + 1.079 / buckets)


def harmonic_estimate(registers):
    """把一串桶值按调和平均折算成元素数。"""
    count = len(registers)
    if count < 1:
        raise ValueError("寄存器序列不能为空")
    total = 0.0
    for value in registers:
        total += 2.0 ** -value
    return alpha_for(count) * count ** 2 / total


def linear_count(buckets, zeros):
    """线性计数：按空桶比例还原元素数，用于桶值很稀的区间。"""
    _require_int(buckets, "桶数")
    _require_int(zeros, "空桶数")
    if buckets < 1:
        raise ValueError("桶数必须为正: %r" % (buckets,))
    if zeros < 1:
        raise ValueError("空桶数必须为正: %r" % (zeros,))
    return buckets * math.log(buckets / zeros)


def large_range_corrected(estimate, hash_bits):
    """大基数修正：估计逼近摘要上界时把调和平均拉回总量空间。"""
    _require_int(hash_bits, "摘要位宽")
    ceiling = float(1 << hash_bits)
    if estimate >= ceiling:
        return ceiling
    return -ceiling * math.log1p(-estimate / ceiling)


def corrected_estimate(raw, buckets, zeros, hash_bits=64):
    """按稀疏程度给调和平均估计套上小基数与大基数两端的修正。"""
    if raw <= 2.5 * buckets and zeros > 0:
        return linear_count(buckets, zeros)
    if raw > (1 << hash_bits) / 30.0:
        return large_range_corrected(raw, hash_bits)
    return raw


def estimate_from_registers(registers, hash_bits=64):
    """按一串桶值直接给估计，供装载快照或手算使用。"""
    values = tuple(registers)
    if not values:
        raise ValueError("寄存器序列不能为空")
    return corrected_estimate(harmonic_estimate(values), len(values),
                              values.count(0), hash_bits)


class HyperLogLog:
    """固定桶数的基数估计器：只增不减的桶值与两端修正的估计。"""

    __slots__ = ("_precision", "_buckets", "_hash_bits", "_hash_fn",
                 "_exact_limit", "_registers", "_exact")

    def __init__(self, precision=DEFAULT_PRECISION, hash_fn=fnv1a64,
                 hash_bits=DEFAULT_HASH_BITS, exact_limit=DEFAULT_EXACT_LIMIT):
        _require_int(hash_bits, "摘要位宽")
        if hash_bits < 8 or hash_bits > 64:
            raise ValueError("摘要位宽必须落在 8 到 64 之间: %r" % (hash_bits,))
        _require_int(precision, "精度")
        if precision < 4 or precision > hash_bits - 2:
            raise ValueError("精度必须落在 4 到 %d 之间: %r"
                             % (hash_bits - 2, precision))
        if not callable(hash_fn):
            raise TypeError("散列函数必须可调用: %r" % (hash_fn,))
        _require_int(exact_limit, "精确区间上限")
        if exact_limit < 0:
            raise ValueError("精确区间上限不能为负: %r" % (exact_limit,))
        self._precision = precision
        self._buckets = 1 << precision
        self._hash_bits = hash_bits
        self._hash_fn = hash_fn
        self._exact_limit = exact_limit
        self._registers = [0] * self._buckets
        self._exact = set() if exact_limit else None

    @classmethod
    def from_registers(cls, registers, precision=DEFAULT_PRECISION,
                       hash_fn=fnv1a64, hash_bits=DEFAULT_HASH_BITS):
        """按已有桶值装配一份快照，用于装载或按寄存器做实验。"""
        sketch = cls(precision, hash_fn, hash_bits, 0)
        values = tuple(registers)
        if len(values) != sketch._buckets:
            raise ValueError("寄存器个数与精度不符: %r" % (len(values),))
        top = sketch.max_rank
        for value in values:
            _require_int(value, "桶值")
            if value < 0 or value > top:
                raise ValueError("桶值必须落在 0 到 %d 之间: %r" % (top, value))
        sketch._registers = list(values)
        return sketch

    def __repr__(self):
        return "HyperLogLog(precision=%d, buckets=%d, estimate=%.1f)" % (
            self._precision, self._buckets, self.estimate())

    @property
    def precision(self):
        return self._precision

    @property
    def buckets(self):
        return self._buckets

    @property
    def hash_bits(self):
        return self._hash_bits

    @property
    def exact_limit(self):
        return self._exact_limit

    @property
    def max_rank(self):
        """桶值可能取到的最大值。"""
        return self._hash_bits - self._precision + 1

    def is_exact(self):
        """去重元素数是否还落在精确区间内。"""
        return self._exact is not None

    def position_of(self, digest):
        """摘要对应的（桶号，桶值）。"""
        _require_int(digest, "摘要")
        if digest < 0 or digest > MASK64:
            raise ValueError("摘要必须落在 64 位无符号范围内: %r" % (digest,))
        digest &= (1 << self._hash_bits) - 1
        index = digest & (self._buckets - 1)
        tail = digest >> self._precision
        rank = leading_zero_rank(tail, self._hash_bits - self._precision)
        return index, rank

    def add(self, key):
        """记录一个元素，返回桶值是否因此变大。"""
        digest = self._hash_fn(key)
        _require_int(digest, "摘要")
        index, rank = self.position_of(digest)
        changed = self._record(index, rank)
        if self._exact is not None:
            self._exact.add(key)
            if len(self._exact) > self._exact_limit:
                self._exact = None
        return changed

    def add_digest(self, digest):
        """按已知摘要记录一个元素；摘要流不再保证小基数精确区间。"""
        index, rank = self.position_of(digest)
        self._exact = None
        return self._record(index, rank)

    def registers(self):
        """全部桶值，按桶号排列。"""
        return tuple(self._registers)

    def zero_buckets(self):
        """还没有元素落入的桶数。"""
        return sum(1 for value in self._registers if value == 0)

    def raw_estimate(self):
        """未做两端修正的调和平均估计。"""
        return harmonic_estimate(self._registers)

    def estimate(self):
        """当前估计：精确区间内是去重个数，其余按桶值修正后给出。"""
        if self._exact is not None:
            return float(len(self._exact))
        return corrected_estimate(self.raw_estimate(), self._buckets,
                                  self.zero_buckets(), self._hash_bits)

    def relative_error(self):
        """理论相对标准误差。"""
        return 1.04 / math.sqrt(self._buckets)

    def merged(self, other):
        """两份同参数快照的并集：桶值取最大，精确区间取并集。"""
        if not isinstance(other, HyperLogLog):
            raise TypeError("只能与同类型快照合并: %r" % (other,))
        if (other._precision != self._precision
                or other._hash_bits != self._hash_bits):
            raise ValueError("两份快照的精度与摘要位宽必须一致")
        result = HyperLogLog(self._precision, self._hash_fn, self._hash_bits,
                             self._exact_limit)
        result._registers = [max(left, right) for left, right
                             in zip(self._registers, other._registers)]
        if self._exact is not None and other._exact is not None:
            group = self._exact | other._exact
            if len(group) <= result._exact_limit:
                result._exact = group
            else:
                result._exact = None
        else:
            result._exact = None
        return result

    def _record(self, index, rank):
        """桶值取最大，返回是否变大。"""
        if rank > self._registers[index]:
            self._registers[index] = rank
            return True
        return False
