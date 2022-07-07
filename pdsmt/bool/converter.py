# coding: utf-8
"""
Some facilities for converting between different forms of Boolean formulas
E.g., tseitin's transformation
"""
from typing import List


def flatter(lst):
    x = []
    for i in lst:
        abs_lst = [abs(j) for j in i]
        x.extend(abs_lst)
    return x


def tseitin(dnf: List[List]):
    """
    Tseitin algorithm: may introduce auxiliary variables
    """
    maxi = max(flatter(dnf))
    next_one = maxi + 1
    ans = []
    for i in dnf:
        ans.append([-1 * i[j] for j in range(len(i))] + [next_one])
        for j in i:
            ans.append([j, -1 * next_one])
        next_one += 1
    return ans


def test():
    dnf = [[-1, -2, 4], [1, -4], [2, -4], [3, 5], [-3, -5]]
    print(tseitin(dnf))


if __name__ == "__main__":
    test()
