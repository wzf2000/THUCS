from itertools import permutations
from collections import defaultdict
from typing import Dict, List, Tuple, Union
from decoder.enigma import Enigma
from decoder.utils import *

class Decoder:
    '''
    Enigma Decoder
    '''
    
    def __init__(self, text: str, cipher: str) -> None:
        assert(len(text) == len(cipher))
        self.text = text
        self.cipher = cipher
        self.chains: List[List[Tuple[str, str, int]]] = []
        self.plugboard_set = []

    def _fetch_chains(self) -> None:
        length = len(self.text)
        edges = [(self.text[i], self.cipher[i], i) for i in range(length)]
        queue = [[(self.text[i], self.cipher[i], i)] for i in range(length)]

        while len(queue) > 0:
            now = queue.pop()
            letters = [x[0] for x in now]
            pos = [x[2] for x in now]

            for x, y, i in edges:
                if i < pos[0] or (i == pos[0] and len(now) == 1):
                    continue

                if now[-1][1] == x:
                    next = (x, y, i)
                elif now[-1][1] == y:
                    next = (y, x, i)
                else:
                    continue

                if next[1] in letters:
                    if next[2] not in pos and letters[0] == next[1]:
                        chain = now + [next]
                        self.chains.append(chain)
                else:
                    queue.append(now + [next])
        
        print(f'fetched {len(self.chains)} chains:')
        for chain in self.chains:
            print(f'length = {len(chain)}, chain = {chain}')

    def _check_rotors_id(self, enigma: Enigma) -> Tuple[int, Union[List[str], None]]:
        all_pos = enigma.generate_all_pos()
        length = len(all_pos)

        plugboard: Dict[str, str] = {}

        def _check_chain(chain: List[Tuple[str, str, int]], init_pos: int) -> int:
            ret = -1
            poss = []
            for num in range(N):
                ch = num2ch(num)
                out = ch
                cnt = 0
                for _, to, offset in chain:
                    enigma.set_pos(all_pos[init_pos + offset - length])
                    out = enigma.forward(out)
                    if out == to:
                        cnt += 1
                if out == ch:
                    if ret < cnt:
                        ret = cnt
                    poss.append(ch)
            if len(poss) == 1:
                ch = poss[0]
                out = ch
                for _, to, offset in chain:
                    enigma.set_pos(all_pos[init_pos + offset - length])
                    out = enigma.forward(out)
                    if out != to:
                        if out in plugboard and plugboard[out] != to or to in plugboard and plugboard[to] != out:
                            return -1
                        plugboard[out] = to
                        plugboard[to] = out
                        if len(plugboard) > 12:
                            return -1
            return ret
        
        ret: List[List[str]] = None
        max_num = 0
        ret_cnt = 0
        for init_pos in range(N ** K - N ** (K - 1)):
            plugboard: Dict[str, str] = {}
            check = True
            num = 0
            for chain in self.chains:
                cnt = _check_chain(chain, init_pos)
                if cnt < 0:
                    check = False
                    break
                else:
                    num += cnt
            
            if not check:
                continue

            if check:
                ret_cnt += 1
                if max_num < num:
                    max_num = num
                    ret = num2list(init_pos)
        print(f'found {ret_cnt} settings')
        return max_num, ret

    def _decode_rotor_set(self) -> None:
        self.max_num = -1
        for rotors_id in permutations(list(range(M)), K):
            max_num, init_position = self._check_rotors_id(Enigma(rotors_id))
            print(f'rotors\' order {rotors_id}, max matched number = {max_num}, setting = {init_position}')
            if max_num > self.max_num:
                self.max_num = max_num
                self.rotors_id = rotors_id
                self.init_position = init_position

        if self.max_num != -1:
            print('Rotor set found!')
            print(f'Rotors\' id: {self.rotors_id}')
            print(f'Rotors\' initial position: {self.init_position}')

    def decode(self) -> None:
        self._fetch_chains()
        self._decode_rotor_set()