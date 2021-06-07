from yoonspeech.speech import YoonSpeech as speech
from yoonspeech.data import YoonObject as object
from yoonspeech.data import YoonDataset as dataset
from yoonspeech.parser import *

phoneme_list = ["aa aa aa", "ae ae ae", "ah ah ah", "ao ao aa", "aw aw aw", "ax ax ah", "ax-h ax ah", "axr er er",  # 8
                "ay ay ay",  # 1
                "b b b", "bcl vcl sil",  # 2
                "ch ch ch",  # 1
                "d d d", "dcl vcl sil", "dh dh dh", "dx dx dx",  # 4
                "eh eh eh", "el el l", "em m m", "en en n", "eng ng ng", "epi epi sil", "er er er", "ey ey ey",  # 8
                "f f f",  # 1
                "g g g", "gcl vcl sil",  # 2
                "h# sil sil", "hh hh hh", "hv hh hh",  # 3
                "ih ih ih", "ix ix ih", "iy iy iy",  # 3
                "jh jh jh",  # 1
                "k  k  k", "kcl cl sil",  # 2
                "l l l",  # 1
                "m m m",  # 1
                "n n n", "ng ng ng", "nx n n",  # 3
                "ow ow ow", "oy oy oy",  # 2
                "p p p", "pau sil sil", "pcl cl sil",  # 3
                "q",  # 1
                "r r r",  # 1
                "s s s", "sh sh sh",  # 2
                "t t t", "tcl cl sil", "th th th",  # 3
                "uh uh uh", "uw uw uw", "ux uw uw",  # 3
                "v v v",  # 1
                "w w w",  # 1
                "y y y",  # 1
                "z z z", "zh zh sh",  # 2
                "sil sil sil"]  # 1

DEFAULT_PHONEME_COUNT = 41
DEFAULT_LIBRISPEECH_COUNT = 40
