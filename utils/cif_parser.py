"""
Parse mmCIF files from PDB_RNA to extract C1' coordinates, full atoms, and derived base pairing.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# C1'-C1' distance thresholds for base pairing (Angstroms)
WC_PAIR_MAX_DIST = 12.0
WC_PAIR_MIN_DIST = 8.0
NONWC_PAIR_MAX_DIST = 14.0
ADJACENT_BACKBONE = 6.5


def _read_cif_blocks(path: Path) -> List[Dict[str, Any]]:
    """Read CIF and return list of block dicts with _atom_site.* arrays."""
    blocks = []
    current = {}
    in_loop = False
    loop_keys = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("data_"):
                if current:
                    blocks.append(current)
                current = {"_name": line[5:].strip()}
                in_loop = False
                continue
            if line.startswith("loop_"):
                in_loop = True
                loop_keys = []
                continue
            if in_loop:
                if line.startswith("_"):
                    loop_keys.append(line.split()[0])
                    if loop_keys and loop_keys[0] not in current:
                        for k in loop_keys:
                            current[k] = []
                    continue
                # Data row
                parts = line.split()
                if not parts or not loop_keys:
                    in_loop = False
                    continue
                # Handle quoted values
                vals = []
                i = 0
                while i < len(parts):
                    if parts[i].startswith("'"):
                        s = parts[i]
                        if s.endswith("'"):
                            vals.append(s[1:-1])
                        else:
                            i += 1
                            while i < len(parts) and not parts[i].endswith("'"):
                                s += " " + parts[i]
                                i += 1
                            if i < len(parts):
                                s += " " + parts[i]
                            vals.append(s[1:-1])
                        i += 1
                        continue
                    vals.append(parts[i])
                    i += 1
                for k, v in zip(loop_keys[: len(vals)], vals):
                    if k in current:
                        try:
                            v = float(v)
                        except ValueError:
                            pass
                        current[k].append(v)
                continue
            if line.startswith("_"):
                idx = line.find(" ")
                if idx > 0:
                    key = line[:idx]
                    val = line[idx + 1 :].strip().strip("'\"")
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                    current[key] = val
    if current:
        blocks.append(current)
    return blocks


def _atom_site_to_arrays(block: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[int]]:
    """Extract label_atom_id, type_symbol, x,y,z, label_asym_id, auth_seq_id from _atom_site loop."""
    prefix = "_atom_site."
    keys = [k for k in block if k.startswith(prefix)]
    if not keys:
        return np.zeros((0, 3)), np.zeros((0, 3)), [], [], []

    base = prefix
    # Find actual key names (some CIFs use auth_comp_id vs label_comp_id etc.)
    group = block.get(base + "group_PDB", block.get(base + "type_symbol", None))
    if group is None:
        for k in keys:
            if "group" in k or "type" in k:
                group = block.get(k, [])
                break
    x = np.array(block.get(base + "Cartn_x", block.get(base + "Cartn_x", [])))
    y = np.array(block.get(base + "Cartn_y", []))
    z = np.array(block.get(base + "Cartn_z", []))
    if isinstance(x, (int, float)):
        x = np.array([x])
    if len(x) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), [], [], []

    coords = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)])

    atom_ids = block.get(base + "label_atom_id", block.get(base + "auth_atom_id", ["X"] * len(x)))
    if isinstance(atom_ids, str):
        atom_ids = [atom_ids]
    comp = block.get(base + "label_comp_id", block.get(base + "auth_comp_id", ["X"] * len(x)))
    if isinstance(comp, str):
        comp = [comp]
    asym = block.get(base + "label_asym_id", block.get(base + "auth_asym_id", ["A"] * len(x)))
    if isinstance(asym, str):
        asym = [asym]
    seq = block.get(base + "label_seq_id", block.get(base + "auth_seq_id", list(range(1, len(x) + 1))))
    if isinstance(seq, (int, float)):
        seq = [seq]
    seq = [int(s) if isinstance(s, (int, float)) else 0 for s in seq]

    return coords, atom_ids, comp, asym, seq


def parse_cif(path: str | Path) -> Dict[str, Any]:
    """
    Parse a single mmCIF file. Returns dict with:
    - c1_prime_coords: dict[chain_id] -> (N, 3) array of C1' coordinates in seq order
    - all_atom_coords: dict[chain_id] -> dict[atom_name] -> (N, 3) for full atoms
    - sequence: dict[chain_id] -> str (one-letter A,U,G,C)
    - base_pairs: list of (chain_i, resi_i, chain_j, resj_j, 'WC'|'nonWC')
    - chain_contacts: set of (chain_i, chain_j) that have residues within 10A
    """
    path = Path(path)
    if not path.exists():
        return {"error": f"File not found: {path}"}

    blocks = _read_cif_blocks(path)
    result = {
        "c1_prime_coords": {},
        "all_atom_coords": {},
        "sequence": {},
        "base_pairs": [],
        "chain_contacts": set(),
        "residue_index": {},
    }

    for block in blocks:
        if "_atom_site.Cartn_x" not in block and "_atom_site.Cartn_x" not in block:
            # Try alternative key names
            for k in block:
                if "Cartn_x" in k:
                    break
            else:
                continue

        prefix = "_atom_site."
        x = block.get(prefix + "Cartn_x", [])
        if not x:
            continue
        if isinstance(x, (int, float)):
            x = [x]
        n = len(x)
        y = block.get(prefix + "Cartn_y", [0] * n)
        z = block.get(prefix + "Cartn_z", [0] * n)
        if isinstance(y, (int, float)):
            y = [y]
        if isinstance(z, (int, float)):
            z = [z]
        coords = np.column_stack([np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)])

        atom_id = block.get(prefix + "label_atom_id", block.get(prefix + "auth_atom_id", ["X"] * n))
        if isinstance(atom_id, str):
            atom_id = [atom_id]
        comp = block.get(prefix + "label_comp_id", block.get(prefix + "auth_comp_id", ["X"] * n))
        if isinstance(comp, str):
            comp = [comp]
        asym = block.get(prefix + "label_asym_id", block.get(prefix + "auth_asym_id", ["A"] * n))
        if isinstance(asym, str):
            asym = [asym]
        seq_id = block.get(prefix + "label_seq_id", block.get(prefix + "auth_seq_id", list(range(1, n + 1))))
        if isinstance(seq_id, (int, float)):
            seq_id = [seq_id]

        comp_to_letter = {"A": "A", "G": "G", "C": "C", "U": "U", "DA": "A", "DG": "G", "DC": "C", "DT": "U"}

        # Group by (asym_id, seq_id) to get per-residue atoms
        from collections import defaultdict

        by_res = defaultdict(list)  # (chain, seq) -> [(atom_name, xyz), ...]
        for i in range(n):
            ch = asym[i] if i < len(asym) else "A"
            sid = int(seq_id[i]) if i < len(seq_id) else i + 1
            aid = atom_id[i] if i < len(atom_id) else "X"
            comp_name = comp[i] if i < len(comp) else "X"
            by_res[(ch, sid)].append((aid.strip(), comp_name, coords[i]))

        for (ch, sid), atoms in by_res.items():
            c1 = [a for a in atoms if a[0] == "C1'"]
            if c1:
                if ch not in result["c1_prime_coords"]:
                    result["c1_prime_coords"][ch] = []
                    result["all_atom_coords"][ch] = defaultdict(list)
                    result["sequence"][ch] = []
                    result["residue_index"][ch] = []
                idx = len(result["c1_prime_coords"][ch])
                result["c1_prime_coords"][ch].append(c1[0][2])
                result["residue_index"][ch].append(sid)
                letter = comp_to_letter.get(comp_name, "N")
                result["sequence"][ch].append(letter)
                for aname, _, xyz in atoms:
                    result["all_atom_coords"][ch][aname].append(xyz)

    # Convert lists to arrays
    for ch in list(result["c1_prime_coords"].keys()):
        result["c1_prime_coords"][ch] = np.array(result["c1_prime_coords"][ch])
        for aname in result["all_atom_coords"][ch]:
            result["all_atom_coords"][ch][aname] = np.array(result["all_atom_coords"][ch][aname])

    # Infer base pairing from C1'-C1' distances (same chain)
    for ch, coords in result["c1_prime_coords"].items():
        N = coords.shape[0]
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(coords[i] - coords[j])
                if d < ADJACENT_BACKBONE and abs(i - j) == 1:
                    continue
                if WC_PAIR_MIN_DIST <= d <= WC_PAIR_MAX_DIST:
                    result["base_pairs"].append((ch, result["residue_index"][ch][i], ch, result["residue_index"][ch][j], "WC"))
                elif d <= NONWC_PAIR_MAX_DIST:
                    result["base_pairs"].append((ch, result["residue_index"][ch][i], ch, result["residue_index"][ch][j], "nonWC"))

    # Chain contacts (different chains, any residue pair < 10A)
    chains = list(result["c1_prime_coords"].keys())
    for i, ch1 in enumerate(chains):
        for ch2 in chains[i + 1 :]:
            c1, c2 = result["c1_prime_coords"][ch1], result["c1_prime_coords"][ch2]
            dmat = np.linalg.norm(c1[:, None, :] - c2[None, :, :], axis=-1)
            if np.any(dmat < 10.0):
                result["chain_contacts"].add((ch1, ch2))

    result["chain_contacts"] = list(result["chain_contacts"])
    return result


def get_c1_prime_for_chain(parsed: Dict[str, Any], chain_id: str) -> Optional[np.ndarray]:
    """Get (N, 3) C1' coords for one chain in order of residue index."""
    if "error" in parsed or chain_id not in parsed.get("c1_prime_coords", {}):
        return None
    return parsed["c1_prime_coords"][chain_id]
