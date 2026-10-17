"""SMEM完整地址需求与cfg_b复合写块接线；不从旧窗口拆出指令价格。"""
from collections import defaultdict


def ordinary_accesses(address_rows_words, width_bytes, direction):
    """按指令→warp→lane保留word、bank与同址关系，地址以32-bit word计。"""
    width = width_bytes//4
    threads = len(address_rows_words[0])
    accesses, distinct = [], set()
    for instruction, row in enumerate(address_rows_words):
        for warp in range(threads//32):
            lanes = [list(range(row[warp*32+lane], row[warp*32+lane]+width))
                     for lane in range(32)]
            owners = defaultdict(list)
            banks = [set() for _ in range(32)]
            for lane, words in enumerate(lanes):
                for word in words:
                    owners[word].append(lane)
                    banks[word % 32].add(word)
                    distinct.add(word)
            accesses.append(dict(instruction=instruction, warp=warp, lane_words=lanes,
                bank_words=[sorted(words) for words in banks],
                same_word_lanes=[dict(word=word, lanes=owners[word])
                                 for word in sorted(owners) if len(owners[word]) > 1]))
    return dict(direction=direction, width_bytes=width_bytes, threads=threads,
        instructions_per_warp=len(address_rows_words),
        warp_instructions=len(address_rows_words)*threads//32,
        requested_bytes=len(address_rows_words)*threads*width_bytes,
        distinct_address_bytes=4*len(distinct), accesses=accesses,
        interpretation='program addresses and bank mapping; no physical service-batch count')


def with_output_write_cost(parameters, *, cycles_per_subtile, frequency_ghz):
    """显式复合STS写块工作假设；替换每8KiB子块的旧写服务，单位ns。"""
    return dict(parameters, epilogue_write_ns=cycles_per_subtile/frequency_ghz)
