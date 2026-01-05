import os
from typing import BinaryIO
from multiprocessing import Pool
import regex as re
from tqdm import tqdm
from collections import defaultdict



# 直接抄的示例代码对训练语料进行分块，以便进行多进程预分词
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))




# 每个进程将要执行的预分词内容
def pre_tokenizer(input_path: str, 
                  start: int, 
                  end: int,
                  pat: re.Pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""),
                  ) -> dict[tuple[bytes], int]:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")

        # Run pre-tokenization on your chunk and store the counts for each pre-token
        chunks = chunk.split("<|endoftext|>")

        block_bytes = {}
        # GPT-2 的 regex 已编译

        for single_chunk in tqdm(chunks, desc=f"{start} --- {end} 块处理中"):
            matchs = pat.finditer(single_chunk)
            for match in matchs:
                # 直接tuple 后实际上存在tuple里面的是int型数据，后续转换成bytes就是：bytes([int])就能转换回去能decode的bytes流了
                # 所以我们加上bytes([])这一步
                match_byte = tuple([bytes([i]) for i in match.group().encode(encoding='utf-8', errors='ignore')])
                block_bytes[match_byte] = block_bytes.get(match_byte, 0) + 1

        return block_bytes




def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # 初始化原始词表和合并记录列表
    vocab_dict = {}
    merge_list = []

    # 添加初始特殊token到词表的最前面
    for i in special_tokens:
        vocab_dict[len(vocab_dict)] = i.encode(encoding='utf-8', errors='ignore')
    
    # 添加初始基础256个tokens
    base_256_bytes = [bytes([code]) for code in range(256)]
    assert len(base_256_bytes) == 256
    for i in base_256_bytes:
        vocab_dict[len(vocab_dict)] = i



    
    # 预分词 Pre-tokenizer
    # 需要多进程进行加速，先对初始文本进行分块
    # 然后分多个进程构建出来训练文本的以下内容
    # dict[Tuple(bytes, ...), int]  ->  出现次数
    # 之后进行合并
    ## Usage
    with open(input_path, "rb") as f:
        num_processes = 16  # 使用多少个进程来进行处理
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # 提前编译正则
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    compiled_pat = re.compile(PAT)

    # 使用进程池
    with Pool(num_processes) as pool:
        results = pool.starmap(pre_tokenizer, [(input_path, start, end, compiled_pat) for start, end in zip(boundaries[:-1], boundaries[1:])])

    # 批量合并结果，使用 defaultdict
    all_bytes = defaultdict(int)
    for block_bytes in tqdm(results, desc="合并所有块"):
        for key, value in block_bytes.items():
            all_bytes[key] += value




    # 完成预分词后开始BPE merge过程 （核心）
    # 在词表数量达到要求前持续重复merge

    # 设置总的迭代次数
    total_iterations = vocab_size - len(vocab_dict)
    # 创建一个tqdm进度条对象
    pbar = tqdm(total=total_iterations)
    merge_pairs = defaultdict(int)
    while len(vocab_dict) < vocab_size:
        pbar.update(1)

        # 初始化merge_pairs表。除了第一次外，之后都靠持续更新，而不是重新遍历
        if len(merge_pairs) == 0:
            for key, value in all_bytes.items():
                for start_byte, end_byte in zip(key[:-1], key[1:]):
                    merge_pairs[tuple(i for i in [start_byte, end_byte])] += value

        sorted_merge_pairs = sorted(
            merge_pairs.items(),
            key=lambda x : (x[1], x[0][0], x[0][1]),
            reverse=True
        )


        # 提取待合并的字符对和其出现次数。并将合并信息和新词记录
        merge_list.append(sorted_merge_pairs[0][0])
        the_new_vocab = sorted_merge_pairs[0][0][0] + sorted_merge_pairs[0][0][1]
        vocab_dict[len(vocab_dict)] = the_new_vocab

        # 快速更新法：
        # 依据删除的词，针对性更新all_bytes和merge_pairs (避免全量遍历)
        # 需要先收集所有包含the_new_vocab的字节元组，然后直接用于修改merge_list
        to_prcess = []
        for byte_tuple, count in all_bytes.items():
            # 检查当前字节原则是否包含待合并的字符对:
            if any(byte_tuple[i] == sorted_merge_pairs[0][0][0] and byte_tuple[i+1] == sorted_merge_pairs[0][0][1]
                   for i in range(len(byte_tuple)-1)):
                to_prcess.append( (byte_tuple, count) )
        # 然后更新all_bytes 和 merge_pairs：
        del merge_pairs[sorted_merge_pairs[0][0]]
        for old_tuple, count in to_prcess:
            del all_bytes[old_tuple]
            new_tuple = []
            to_delete_tuple = []
            to_add_tuple = []
            i = 0
            while True:
                if i >= len(old_tuple):
                    break
                if i<len(old_tuple)-1 and old_tuple[i] + old_tuple[i+1] == the_new_vocab:
                    # 先记载需要删除的组合：
                    if i != 0:  # 往前组合
                        to_delete_tuple.append(tuple(j for j in [old_tuple[i-1], old_tuple[i]]))
                        to_add_tuple.append(tuple(j for j in [old_tuple[i-1], the_new_vocab]))
                    if i+2 < len(old_tuple):
                        to_delete_tuple.append(tuple(j for j in [old_tuple[i+1], old_tuple[i+2]]))
                        to_add_tuple.append(tuple(j for j in [the_new_vocab, old_tuple[i+2]]))
                    new_tuple.append(the_new_vocab)
                    i += 2
                else:
                    new_tuple.append(old_tuple[i])
                    i += 1
            # 组装出新的new_tuple后重新加入字典中
            all_bytes[tuple(new_tuple)] = count

            for cur_delete_tuple in to_delete_tuple:
                merge_pairs[cur_delete_tuple] -= count
                if merge_pairs[cur_delete_tuple] == 0:
                    del merge_pairs[cur_delete_tuple]
            for cur_add_tuple in to_add_tuple:
                merge_pairs[cur_add_tuple] += count





        # 遍历式更新方法，时间复杂度太高了，太慢了
        # new_bytes = defaultdict(int)
        # for key, value in all_bytes.items():
        #     cache = []
        #     for start_byte, end_byte in zip(key[:-1], key[1:]):
        #         if start_byte + end_byte == the_new_vocab:
        #             i = 0
        #             while True:
        #                 if i >= len(key):
        #                     break
        #                 if i<len(key)-1 and key[i] + key[i+1] == the_new_vocab:
        #                     cache.append(the_new_vocab)
        #                     i += 2
        #                 else:
        #                     cache.append(key[i])
        #                     i += 1
        #             new_bytes[tuple(cache)] = value
        #             break
        #     if len(cache) == 0:
        #         new_bytes[key] = value
        # all_bytes = new_bytes


    return vocab_dict, merge_list

def save_vocab_and_merges(vocab_dict: dict,
                          merge_list: list[tuple[bytes, bytes]],
                          vocab_file: str,
                          merges_file: str):
    """
    保存 BPE 训练结果到文件：
    vocab_file: 词表文件，格式 <token_id> <token_bytes_hex>
    merges_file: merge 规则文件，格式 <left_bytes_hex> <right_bytes_hex>
    """

    # 写 vocab_file
    with open(vocab_file, "w", encoding="utf-8") as vf:
        # vocab_dict: {token_bytes: token_id}
        # 注意：token_bytes 是 bytes
        for token_id, token_bytes in vocab_dict.items():
            token_hex = token_bytes.hex()
            vf.write(f"{token_id} {token_hex}\n")

    # 写 merges_file
    with open(merges_file, "w", encoding="utf-8") as mf:
        # merge_list: [(left_bytes, right_bytes), ...]
        for left, right in merge_list:
            mf.write(f"{left.hex()} {right.hex()}\n")


if __name__=='__main__':
    vocab_dict, merge_list = run_train_bpe(input_path='/home/bianyuhan/LLM Learning/cs336/data/TinyStoriesV2-GPT4-train.txt',
                                            vocab_size=20000,
                                            special_tokens=['<|endoftext|>'])
    print(vocab_dict)

    save_vocab_and_merges(
        vocab_dict,
        merge_list,
        vocab_file="/home/bianyuhan/LLM Learning/cs336/cs336-1/cs336_basics/vocab.txt",
        merges_file="/home/bianyuhan/LLM Learning/cs336/cs336-1/cs336_basics/merges.txt"
    )

