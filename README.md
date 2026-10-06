# pairwise-hyperloglog

只依赖 Python 标准库的基数估计内核：元素与散列函数都由调用方给出，同一串元素
在任何进程里都得到同样的寄存器与估计值，不使用随机数、时钟、网络或 I/O。

- `hyperloglog/core.py` — 内核：前导零计数与估计常量，分桶寄存器（写入元素、
  按摘要写入、查桶号与桶值、读全部桶值、空桶数、未修正与修正后的估计、理论相对
  误差、合并两份快照），以及从小基数线性计数到大基数上界修正的两端处理。
- `tests/test_core.py` — 验收用例。

## 运行测试

在项目根目录执行：

```
python3 -m unittest discover -s tests -v
```

Windows 上如果没有 `python3`，可用：

```
python -m unittest discover -s tests -v
```
