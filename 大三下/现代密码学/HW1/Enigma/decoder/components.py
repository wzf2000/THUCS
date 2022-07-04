from typing import List, Tuple, Union
from decoder.utils import *

class Rotor:
    '''
    Settings and APIs for Rotors
    '''
    rotors_init = rotors_setting
    rotors_carry = [
        'R',
        'F',
        'W',
        # 'K',
        # 'A'
    ]
    
    def __init__(self, id: int, ring_setting: int, pos: int = 0) -> None:
        assert(id < len(Rotor.rotors_init))
        self.rotor = [ ch2num(ch) for ch in Rotor.rotors_init[id] ]
        self.rotor_rev = [ Rotor.rotors_init[id].index(num2ch(i)) for i in range(N) ]
        self.rotor_carry = ch2num(Rotor.rotors_carry[id])
        self.ring_setting = ring_setting
        self.pos = pos

    def set(self, pos=0) -> None:
        self.pos = pos

    def forward(self, num: int) -> int:
        delta = sub(self.pos, self.ring_setting)
        return sub(self.rotor[add(num, delta)], delta)

    def backward(self, num: int) -> int:
        delta = sub(self.pos, self.ring_setting)
        return sub(self.rotor_rev[add(num, delta)], delta)

    def rotate(self) -> bool:
        self.pos = next(self.pos)
        return self.pos == self.rotor_carry

class Reflector:
    '''
    Settings and APIs for Reflector
    '''
    reflector_init = 'YRUHQSLDPXNGOKMIEBFZCWVJAT'

    def __init__(self) -> None:
        self.reflector = [
            ord(ch) - ord('A') for ch in Reflector.reflector_init
        ]

    def forward(self, num: int) -> int:
        return self.reflector[num]

reflector = Reflector()

class Plugboard:
    '''
    Settings and APIs for Plugboard
    '''

    def __init__(self, swap_pairs: List[Tuple[Union[int, str], Union[int, str]]]) -> None:
        assert(len(swap_pairs) <= 6)
        self.plugboard = list(range(N))
        for x, y in swap_pairs:
            if isinstance(x, str):
                x = ch2num(x)
            if isinstance(y, str):
                y = ch2num(y)
            self.plugboard[x], self.plugboard[y] = y, x

    def forward(self, num: int) -> int:
        return self.plugboard[num]
