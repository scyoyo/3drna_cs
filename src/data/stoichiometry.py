"""
Parse stoichiometry and extract chain sequences from all_sequences (FASTA format).
Stoichiometry format: {chain:number}; e.g. "A:1;B:2" = chain A once, chain B twice.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .featurizer import encode_sequence


def parse_fasta(all_sequences: str) -> Dict[str, str]:
    """
    Parse FASTA string (from all_sequences column) into dict chain_id -> sequence.
    Header format may include "Chains" and "|" delimiters; chain id in chain=X tag.
    """
    result = {}
    current_id = None
    current_seq = []

    for line in all_sequences.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                result[current_id] = "".join(current_seq).upper().replace("T", "U")
            # Parse header for chain id: chain=A or similar
            current_id = "A"  # default
            for part in line[1:].split("|"):
                part = part.strip()
                if part.lower().startswith("chain="):
                    current_id = part.split("=", 1)[1].strip()
                    break
            # Sometimes header is just ">A" or ">Chain A"
            if current_id == "A" and "chain" not in line.lower():
                m = re.search(r">\s*(\w+)", line)
                if m:
                    current_id = m.group(1)
            current_seq = []
        else:
            current_seq.append(line)

    if current_id is not None:
        result[current_id] = "".join(current_seq).upper().replace("T", "U")
    return result


def parse_stoichiometry(stoi: str) -> List[Tuple[str, int]]:
    """
    Parse stoichiometry string. Returns list of (chain_id, count).
    Example: "A:1;B:2" -> [("A", 1), ("B", 2)]
    """
    if not stoi or not str(stoi).strip():
        return [("A", 1)]
    result = []
    for part in str(stoi).split(";"):
        part = part.strip()
        if ":" in part:
            ch, num = part.split(":", 1)
            result.append((ch.strip(), int(num)))
        else:
            result.append((part, 1))
    return result


def get_chain_sequences(
    all_sequences: str,
    stoichiometry: str,
) -> Tuple[str, List[Tuple[str, int]], Dict[str, str]]:
    """
    From all_sequences (FASTA) and stoichiometry, return:
    - concatenated_sequence: full target sequence (chains in stoichiometry order, repeated by copy count)
    - chain_copies: list of (chain_id, copy_count) for each unique chain in order
    - chain_to_seq: dict chain_id -> single chain sequence

    So concatenated_sequence = seq(chain1)*copy1 + seq(chain2)*copy2 + ...
    """
    chain_to_seq = parse_fasta(all_sequences)
    stoi = parse_stoichiometry(stoichiometry)
    parts = []
    for ch, count in stoi:
        seq = chain_to_seq.get(ch, "")
        for _ in range(count):
            parts.append(seq)
    return "".join(parts), stoi, chain_to_seq


def build_chain_index(stoi: List[Tuple[str, int]], chain_to_seq: Dict[str, str]) -> List[Tuple[int, int, str, int]]:
    """
    Build residue index to (start, end, chain_id, copy_number).
    Returns list of (start_0based, end_0based, chain_id, copy_number) for each segment.
    """
    segments = []
    pos = 0
    for ch, count in stoi:
        seq = chain_to_seq.get(ch, "")
        L = len(seq)
        for copy_idx in range(count):
            segments.append((pos, pos + L, ch, copy_idx + 1))
            pos += L
    return segments
