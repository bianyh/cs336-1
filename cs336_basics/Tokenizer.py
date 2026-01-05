from typing import Iterable
from typing import Iterator
import regex as re
from tqdm import tqdm
import json
import os
from collections import defaultdict


def encode_pre_tokenizer(input_text: str, 
                        pat: re.Pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""),
                        special_tokens: list[str] = None,
                        ) -> tuple[bytes] | bytes:
    input_text = input_text.replace("\r\n", "\n").replace("\r", "\n")

    # # 在换行处添加 <|endoftext|> 以让模型学习句子结束
    # if special_tokens and "<|endoftext|>" in special_tokens:
    #     input_text = input_text.replace("\n", "<|endoftext|>\n")

    block_bytes = []

    if special_tokens:
        # 构造正则：捕获所有 special_tokens
        escaped_tokens = [re.escape(tok) for tok in special_tokens]
        # escaped_tokens.append(re.escape('\n'))
        split_pattern = re.compile("(" + "|".join(escaped_tokens) + ")")
        chunks = split_pattern.split(input_text)
    else:
        chunks = [input_text]

    # for single_chunk in tqdm(chunks, desc="预分词中"):
    for single_chunk in chunks:
        if single_chunk in (special_tokens or []):
            # 如果是 special_token，直接保留
            block_bytes.append(single_chunk.encode("utf-8", errors="ignore"))
        else:
            # 正常分词
            # block_bytes.append(tuple([bytes([i]) for i in single_chunk.encode("utf-8", errors="ignore")]))
            matchs = pat.finditer(single_chunk)
            for match in matchs:
                match_byte = tuple([bytes([i]) for i in match.group().encode("utf-8", errors="ignore")])
                block_bytes.append(match_byte)

    return tuple(block_bytes)


def load_vocab_and_merges(vocab_file: str, merges_file: str):
    vocab_dict = {}


    with open(vocab_file, "r", encoding="utf-8") as vf:
        for line in vf:
            line = line.strip()
            if not line:
                continue
            token_id_str, token_hex = line.split()
            token_id = int(token_id_str)
            token_bytes = bytes.fromhex(token_hex)
            vocab_dict[token_id] = token_bytes

    merge_list = []
    with open(merges_file, "r", encoding="utf-8") as mf:
        for line in mf:
            line = line.strip()
            if not line:
                continue
            left_hex, right_hex = line.split()
            left = bytes.fromhex(left_hex)
            right = bytes.fromhex(right_hex)
            merge_list.append((left, right))

    return vocab_dict, merge_list


class MyTokenizer():
    def __init__(self, vocab: dict[int, bytes] = None,
                 merges: list[tuple[bytes, bytes]] = None,
                 special_tokens: list[str] | None = None,
                 ):
        self.vocab = vocab
        self.token_to_ids_vocab = {}
        for ids, bytes in self.vocab.items():
            self.token_to_ids_vocab[bytes] = ids


        self.merges = merges
        self.merges_dict = defaultdict()
        for i in range(len(self.merges)):
            self.merges_dict[self.merges[i]] = i


        if special_tokens is not None:
            self.special_tokens = sorted(special_tokens,
                                         key=lambda x: len(x),
                                         reverse=True)
            self.special_tokens_bytes = [s.encode(encoding='utf-8') for s in self.special_tokens]
        else:
            self.special_tokens_bytes = []
            self.special_tokens = []
        



    @classmethod
    def from_files(cls,
                   vocab_filepath,
                   merges_filepath,
                   special_tokens=None
                   ):
        vocab_dict, merge_list = load_vocab_and_merges(vocab_file=vocab_filepath,
                                                       merges_file=merges_filepath,)
        return cls(vocab_dict, merge_list, special_tokens)


    # 很快会发现：
    # encode 阶段其实更想要 bytes -> token_id 的反查表
    # 所以在init的时候直接把这一步做好
    def encode(self, text:str):
        # 首先一样要做预分词
        # dict[tuple[bytes], int]
        all_bytes = encode_pre_tokenizer(input_text=text, special_tokens=self.special_tokens)

        # 对每个字节流运用merge规则似乎有些太慢了
        ids = []
        # for byte in tqdm(all_bytes):
        for byte in all_bytes:
            if byte in self.special_tokens_bytes:
                ids.append(self.token_to_ids_vocab[byte])
                continue
            for merge_pattern in self.merges:
                if merge_pattern[0] not in byte or merge_pattern[1] not in byte:
                    continue
                new_bytes = []
                i = 0
                while True:
                    if i >= len(byte):
                        break
                    if i < len(byte)-1 and byte[i] == merge_pattern[0] and byte[i+1] == merge_pattern[1]:
                        new_bytes.append(merge_pattern[0] + merge_pattern[1])
                        i += 2
                    else:
                        new_bytes.append(byte[i])
                        i += 1
                byte = new_bytes
            for b in byte:
                ids.append(self.token_to_ids_vocab[b])
        return ids

        


    def encode_iterable(self,
                        iterable:Iterable[str]
                        )->Iterator[int]:
        for text in iterable: 
            ids = self.encode(text) 
            for token_id in ids: 
                yield token_id

                
        
    # 解码操作
    def decode(self, ids: list[int]) -> str:
        bytes_sum = bytes([])
        for i in ids:
            bytes_sum += self.vocab[i]
        return bytes_sum.decode(encoding='utf-8', errors="replace")


        
if __name__=="__main__":
    # vocab_dict, merge_list = run_train_bpe(input_path='/home/bianyuhan/LLM Learning/cs336/data/TinyStoriesV2-GPT4-valid.txt',
    #                                     vocab_size=6000,
    #                                     special_tokens=['<|endoftext|>'])
    


    # tokenizer = MyTokenizer(vocab=vocab_dict,
    #                         merges=merge_list,
    #                         special_tokens=['<|endoftext|>'])
    
    tokenizer = MyTokenizer.from_files(vocab_filepath="/home/bianyuhan/LLM Learning/cs336/cs336-1/cs336_basics/vocab.txt",
                                       merges_filepath="/home/bianyuhan/LLM Learning/cs336/cs336-1/cs336_basics/merges.txt",
                                       special_tokens=['<|endoftext|>'])




    strs = 'I love y<|endoftext|>ou!🙃'
#     strs = '''<|endoftext|>

# '''
    # strs = ""
    # encode
    ids = tokenizer.encode(strs)
    print(ids)

    print(tokenizer.decode(ids))

    assert strs == tokenizer.decode(ids)

    # texts = ["hello", "world", "🙃"] 
    # for tid in tokenizer.encode_iterable(texts): 
    #     print(tokenizer.decode([tid]))

