text = 'BHUILOPALOPBJXFCE'
cipher = 'PBDPPXJOMEXLFOLGA'

from decoder.utils import *
from decoder.decoder import Decoder
from decoder.enigma import Enigma

# enigma = Enigma((0, 2, 1), [('Q', 'U'), ('B', 'W'), ('T', 'V'), ('I', 'F'), ('O', 'K'), ('P', 'Y')], [ch2num('A'), ch2num('A'), ch2num('A')])
# print(enigma.run(text))

decoder = Decoder(text, cipher)
decoder.decode()
