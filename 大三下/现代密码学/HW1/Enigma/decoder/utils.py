from typing import List
N = 26
rotors_setting = [
    'EKMFLGDQVZNTOWYHXUSPAIBRCJ',
    'AJDKSIRUXBLHWTMCQGZNPYFVOE',
    'BDFHJLCPRTXVZNYEIWGAKMUSQO',
    # 'ESOVPZJAYQUIRHXLNFTGKDCMWB',
    # 'VZBRGITYUPSDNHLXAWMJQOFECK'
]
M = len(rotors_setting)
K = 3

def ch2num(ch: str) -> int:
    return ord(ch) - ord('A')

def num2ch(num: int) -> str:
    return chr(ord('A') + num)

def num2list(num: int) -> List[str]:
    ret: List[str] = []
    for _ in range(K):
        ret.append(num2ch(num % N))
        num //= N
    return ret

def add(x: int, y: int) -> int:
    return (x + y) % N

def sub(x: int, y: int) -> int:
    return (x - y + N) % N

def next(x: int) -> int:
    return add(x, 1)
