from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from phonemizer import phonemize
import sys
# print(EspeakBackend.supported_languages())
lang="en-us"

lines = []
sep = Separator(phone=' ', syllable='', word='')
for line in sys.stdin:
    line = line.strip()
    lines.append(line)
    # print(line)
print(lines)

phones = [ ph.strip() for ph in phonemize(lines, language=lang, separator=sep, language_switch="remove-flags")]
print(phones)