# -*- coding: utf-8 -*-
# Zagier 验证脚本的单元测试: 先于实现编写 (TDD)
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))   # 使 verify 可被导入
from verify import zagier_map, S, fixed_points, two_square_rep, check_prime


def test_S_13():
    # p=13 的解集 S = {(x,y,z): x^2+4yz=13} 恰为三个元素
    assert sorted(S(13)) == [(1, 1, 3), (1, 3, 1), (3, 1, 1)]


def test_involution_p13():
    # Zagier 映射是 S(13) 上的对合: 像在集合内, 二次作用回到自身
    s = set(S(13))
    for t in s:
        assert zagier_map(t) in s
        assert zagier_map(zagier_map(t)) == t
    assert zagier_map((1, 3, 1)) == (3, 1, 1)      # 互换对
    assert zagier_map((1, 1, 3)) == (1, 1, 3)      # 不动点


def test_fixed_point_unique():
    # 对若干 p≡1(mod 4): Zagier 对合的不动点唯一且为 (1,1,(p-1)/4)
    for p in (5, 13, 29, 97, 101):
        assert fixed_points(p) == [(1, 1, (p - 1) // 4)]


def test_two_square():
    # 两平方和表示存在且正确
    assert two_square_rep(13) in {(2, 3), (3, 2)}
    a, b = two_square_rep(1000033)                  # 10^6+33, 素数且 ≡1 (mod 4)
    assert a * a + b * b == 1000033


def test_check_prime_full():
    # 全量检查接口: |S|, 表示均正确
    ok, info = check_prime(13, full=True)
    assert ok and info["n_S"] == 3 and info["rep"][0] ** 2 + info["rep"][1] ** 2 == 13
