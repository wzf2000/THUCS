from typing import List, Tuple, Union
from decoder.components import Rotor, reflector, Plugboard
from decoder.utils import *

class Enigma:
    '''
    Settings and APIs for whole Enigma
    '''

    ring_settings = ['G', 'Y', 'E']
    def __init__(self, rotors_id: List[int], swap_pairs: List[Tuple[Union[int, str], Union[int, str]]] = [], rotors_pos: Union[List[int], None] = None) -> None:
        if rotors_pos == None:
            rotors_pos = [0] * len(rotors_id)
        assert(len(rotors_id) == len(rotors_pos) and len(rotors_pos) == len(Enigma.ring_settings))
        self.rotors = [Rotor(id, ch2num(ring_setting), pos) for id, pos, ring_setting in zip(rotors_id, rotors_pos, Enigma.ring_settings)]
        self.reflector = reflector
        self.plugboard = Plugboard(swap_pairs)
    
    def set_pos(self, rotors_pos: Union[List[int], None] = None) -> None:
        if rotors_pos == None:
            rotors_pos = [0] * len(self.rotors)
        assert(len(rotors_pos) == len(self.rotors))
        for pos, rotor in zip(rotors_pos, self.rotors):
            rotor.set(pos)

    def get_pos(self) -> List[int]:
        return [rotor.pos for rotor in self.rotors]

    def step(self) -> None:
        rotate = True
        i = len(self.rotors) - 1
        while rotate == True and i >= 0:
            rotate = self.rotors[i].rotate()
            i -= 1
            if i > 0 and self.rotors[i].rotor_carry == next(self.rotors[i].pos):
                rotate = True

    def generate_all_pos(self) -> List[List[int]]:
        init = self.get_pos()
        ret = [init]
        self.step()
        now = self.get_pos()
        while now != init:
            ret.append(now)
            self.step()
            now = self.get_pos()
        return ret

    def forward(self, ch: str) -> str:
        self.step()
        num = ch2num(ch)
        num = self.plugboard.forward(num)
        for rotor in self.rotors[::-1]:
            num = rotor.forward(num)
        num = self.reflector.forward(num)
        for rotor in self.rotors:
            num = rotor.backward(num)
        num = self.plugboard.forward(num)
        return num2ch(num)
    
    def run(self, text: str) -> str:
        ret = ''
        for ch in text:
            ret += self.forward(ch)
        return ret
