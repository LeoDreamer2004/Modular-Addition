from modular_add.data import AlgorithmDataTokenizer


def test_tokenizer_len():
    modular = 13
    tokenizer = AlgorithmDataTokenizer(modular)
    assert len(tokenizer) == modular + 3


def test_tokenize():
    tokenizer = AlgorithmDataTokenizer(13)
    eq = "1 + 2 = 3"
    encode = tokenizer.encode(eq)
    assert tokenizer.decode(encode) == eq


def test_tokenize_with_eos():
    tokenizer = AlgorithmDataTokenizer(13)
    eq = "1 + 2 = 3 <eos>"
    encode = tokenizer.encode(eq)
    assert tokenizer.decode(encode) == eq
