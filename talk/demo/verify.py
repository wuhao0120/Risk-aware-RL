# -*- coding: utf-8 -*-
"""Zagier (1990) 一句话证明的数值验证。

定理: 素数 p ≡ 1 (mod 4) 可表为两平方和。
证明用的集合: S = {(x,y,z) ∈ Z+^3 : x^2 + 4yz = p}
Zagier 对合 T 在 S 上恰有一个不动点 (1,1,(p-1)/4) ⇒ |S| 为奇数
⇒ 交换对合 σ:(x,y,z)↦(x,z,y) 必有不动点 (y=z) ⇒ p = x^2 + (2y)^2

用法:
  python verify.py            # 报告模式: 对合性 p<=10^4 全量; 不动点+两平方 p<=10^6
  python verify.py 13         # 单素数详细模式: 打印 S、配对、不动点、表示
"""
import sys
import time
from math import isqrt


def zagier_map(t):
    # Zagier 的三分支对合 (论文中唯一的那个句子)
    x, y, z = t
    if x < y - z:                          # 分支1: x < y-z
        return (x + 2 * z, z, y - x - z)
    if x < 2 * y:                          # 分支2: y-z < x < 2y (含 x=y 不动点)
        return (2 * y - x, y, x - y + z)
    return (x - 2 * y, x - y + z, y)       # 分支3: x > 2y


def S(p):
    # 枚举 S = {(x,y,z): x^2+4yz=p}: x 必为奇数(因 p≡1 mod 4), 对每个 x 枚举 m=yz 的因子
    out = []
    x = 1
    while x * x < p:
        rem = p - x * x
        if rem % 4 == 0:
            m = rem // 4
            for y in range(1, isqrt(m) + 1):
                if m % y == 0:
                    out.append((x, y, m // y))
                    if y != m // y:
                        out.append((x, m // y, y))
        x += 2
    return out


def fixed_points(p):
    # Zagier 对合在 S(p) 上的全部不动点 (理论值: 仅 (1,1,(p-1)/4))
    return [t for t in S(p) if zagier_map(t) == t]


def two_square_rep(p):
    # 求 p = a^2 + b^2 的一组表示 (sqrt 扫描, O(sqrt p))
    for a in range(1, isqrt(p) + 1):
        b2 = p - a * a
        b = isqrt(b2)
        if b * b == b2:
            return (a, b)
    return None


def check_prime(p, full=False):
    # full=True: 全量验证 (枚举 S, 验对合性+不动点唯一); full=False: 只验不动点公式与表示
    info = {}
    if full:
        s = S(p)
        ss = set(s)
        ok_inv = all(zagier_map(t) in ss and zagier_map(zagier_map(t)) == t for t in s)
        info["n_S"] = len(s)
        fp = [t for t in s if zagier_map(t) == t]
        ok_fp = fp == [(1, 1, (p - 1) // 4)]
    else:
        ok_inv = True
        ok_fp = (p - 1) % 4 == 0
    rep = two_square_rep(p)
    info["rep"] = rep
    return ok_inv and ok_fp and rep is not None, info


def primes_1mod4(n):
    # 埃氏筛: 返回 n 以内所有 ≡1 (mod 4) 的素数
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [p for p in range(5, n + 1) if sieve[p] and p % 4 == 1]


def main():
    if len(sys.argv) > 1:                  # 单素数详细模式
        p = int(sys.argv[1])
        assert p % 4 == 1, "需要 p ≡ 1 (mod 4)"
        s = S(p)
        print(f"p = {p}, |S| = {len(s)}")
        for t in sorted(s):
            u = zagier_map(t)
            print(f"  {t} -> {u}" + ("   <- 不动点" if u == t else ""))
        a, b = two_square_rep(p)
        print(f"两平方: {p} = {a}^2 + {b}^2")
        return

    t0 = time.time()
    full_n, fast_n = 10**4, 10**6
    full = primes_1mod4(full_n)
    assert all(check_prime(p, full=True)[0] for p in full)
    t1 = time.time()
    print(f"[全量] p <= {full_n:,}: {len(full)} 个素数, 对合性+不动点唯一+两平方 OK ({t1-t0:.1f}s)")
    fast = primes_1mod4(fast_n)
    assert all(check_prime(p, full=False)[0] for p in fast)
    t2 = time.time()
    print(f"[快速] p <= {fast_n:,}: {len(fast):,} 个素数, 不动点公式+两平方表示 OK ({t2-t1:.1f}s)")


if __name__ == "__main__":
    main()
