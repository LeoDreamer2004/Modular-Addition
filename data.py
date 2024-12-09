# We focus on x + y (mod p), tokenize?

import itertools

MODULUS = 7
NUMS = list(range(100))
EPS_TOKEN = "<eps>"
OP_TOKENS = ["+", "="]

class DataTokenizer:
    
    @staticmethod
    def make_data():
        eqs = []
        for a in NUMS:
            for b in NUMS:
                c = (a + b) % MODULUS
                eqs.append(f"{a} + {b} = {c}")
        return eqs

    @staticmethod
    def tokens():
        return list(map(str, range(10))) + OP_TOKENS + [EPS_TOKEN]


class DataSet:
    pass
