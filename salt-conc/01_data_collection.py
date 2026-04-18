"""
=============================================================================
CrystalNet — Stage 1: Data Collection from PDB
=============================================================================
This script:
  1. Queries PDB for protein structures with crystallization metadata
  2. Parses REMARK 280 to extract salt/ion concentration
  3. Fetches FASTA sequences for each PDB entry
  4. Saves a clean dataset: (pdb_id, sequence, salt_concentration)

Run time: ~30-60 min depending on network (fetches thousands of PDB entries)
Output:  crystallization_dataset.csv
=============================================================================
"""

import requests
import re
import time
import csv
import json
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Search PDB for entries with crystallization conditions
# ─────────────────────────────────────────────────────────────────────────────
# PDB's search API lets us find structures solved by X-ray diffraction
# that have REMARK 280 (crystallization conditions) populated.

def search_pdb_ids(max_results=30000):
    import requests
    
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    
    all_ids = []
    batch_size = 1000

    print(f"Fetching up to {max_results} PDB IDs using pagination...")

    for start in range(0, max_results, batch_size):
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "exptl.method",
                            "operator": "exact_match",
                            "value": "X-RAY DIFFRACTION"
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "entity_poly.rcsb_entity_polymer_type",
                            "operator": "exact_match",
                            "value": "Protein"
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.resolution_combined",
                            "operator": "less_or_equal",
                            "value": 3.0
                        }
                    }
                ]
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": start, "rows": batch_size}
            }
        }

        resp = requests.post(search_url, json=query)
        if resp.status_code != 200:
            print("Error:", resp.text)
            break

        data = resp.json()
        ids = [r["identifier"] for r in data.get("result_set", [])]

        if not ids:
            break

        all_ids.extend(ids)
        print(f"Collected {len(all_ids)} IDs...")

    return list(set(all_ids))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Parse REMARK 280 from PDB files for salt concentration
# ─────────────────────────────────────────────────────────────────────────────

def fetch_remark_280(pdb_id):
    """
    Download the PDB file header and extract REMARK 280 lines.
    REMARK 280 contains free-text crystallization conditions.
    """
    url = f"https://files.rcsb.org/header/{pdb_id.upper()}.pdb"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        lines = resp.text.split("\n")
        remark_280 = []
        for line in lines:
            if line.startswith("REMARK 280"):
                remark_280.append(line[11:].strip())  # text after "REMARK 280 "
        return " ".join(remark_280) if remark_280 else None
    except Exception:
        return None


def extract_salt_concentration(remark_text):
    """
    Parse salt/ion concentration from REMARK 280 free text.
    
    Common patterns in REMARK 280:
      - "0.2 M SODIUM CHLORIDE"
      - "200 MM AMMONIUM SULFATE"
      - "0.1M NACL"
      - "100 mM NaCl"
    
    We extract the numeric concentration and normalize to Molar (M).
    We look for common salt names and their concentrations.
    """
    if not remark_text:
        return None, None

    text = remark_text.upper()

    # Common salts in crystallization
    salt_names = [
        "SODIUM CHLORIDE", "NACL",
        "AMMONIUM SULFATE", "AMMONIUM SULPHATE", "(NH4)2SO4",
        "POTASSIUM CHLORIDE", "KCL",
        "MAGNESIUM CHLORIDE", "MGCL2", "MGCL",
        "LITHIUM SULFATE", "LITHIUM SULPHATE", "LI2SO4",
        "CALCIUM CHLORIDE", "CACL2",
        "SODIUM ACETATE", "NAAC", "NA ACETATE", "NAOAC",
        "AMMONIUM CHLORIDE", "NH4CL",
        "SODIUM SULFATE", "SODIUM SULPHATE", "NA2SO4",
        "ZINC ACETATE",
        "SODIUM FORMATE",
        "POTASSIUM PHOSPHATE",
        "SODIUM CITRATE",
        "AMMONIUM ACETATE",
        "SODIUM MALONATE",
    ]

    # Pattern: number + unit + salt_name  OR  salt_name + number + unit
    # Units: M, mM, MM (PDB often uses MM for millimolar)
    for salt in salt_names:
        if salt not in text:
            continue

        # Try to find concentration near the salt name
        # Pattern 1: "0.2 M SODIUM CHLORIDE" or "0.2M NACL"
        pattern1 = rf'(\d+\.?\d*)\s*(M\b|MM\b|mM\b)\s+{re.escape(salt)}'
        match = re.search(pattern1, text)
        if match:
            value = float(match.group(1))
            unit = match.group(2).upper()
            if unit in ("MM", "MM"):
                value /= 1000.0  # convert mM to M
            return round(value, 4), salt

        # Pattern 2: "SODIUM CHLORIDE 0.2 M"
        pattern2 = rf'{re.escape(salt)}\s+(\d+\.?\d*)\s*(M\b|MM\b|mM\b)'
        match = re.search(pattern2, text)
        if match:
            value = float(match.group(1))
            unit = match.group(2).upper()
            if unit in ("MM", "MM"):
                value /= 1000.0
            return round(value, 4), salt

        # Pattern 3: number immediately before salt without explicit unit
        # e.g., "200 AMMONIUM SULFATE" — assume mM if >= 10, M if < 10
        pattern3 = rf'(\d+\.?\d*)\s+{re.escape(salt)}'
        match = re.search(pattern3, text)
        if match:
            value = float(match.group(1))
            if value >= 10:
                value /= 1000.0  # likely mM
            return round(value, 4), salt

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Fetch FASTA sequence for each PDB entry
# ─────────────────────────────────────────────────────────────────────────────

def fetch_fasta_sequence(pdb_id):
    """
    Fetch the protein sequence for a PDB entry from RCSB FASTA endpoint.
    Returns the sequence of the first chain (entity 1).
    """
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}/download"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        lines = resp.text.strip().split("\n")
        # Take the first chain's sequence
        seq_lines = []
        found_header = False
        for line in lines:
            if line.startswith(">"):
                if found_header:
                    break  # second chain, stop
                found_header = True
                continue
            seq_lines.append(line.strip())
        sequence = "".join(seq_lines)
        # Basic validation
        if len(sequence) < 30:  # too short, skip
            return None
        if any(c not in "ACDEFGHIKLMNPQRSTVWY" for c in sequence):
            # non-standard amino acids — clean them
            sequence = "".join(c for c in sequence if c in "ACDEFGHIKLMNPQRSTVWY")
        return sequence if len(sequence) >= 30 else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Main pipeline — collect everything into a dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(max_search=30000, output_file="crystallization_dataset.csv"):
    """
    Full pipeline:
      1. Get PDB IDs
      2. For each, fetch REMARK 280, extract salt concentration
      3. For successful extractions, fetch FASTA sequence
      4. Save to CSV
    """
    pdb_ids = search_pdb_ids(max_results=max_search)

    records = []
    failed = 0
    no_salt = 0

    print(f"\nProcessing {len(pdb_ids)} entries...")
    print("This will take a while. Progress updates every 100 entries.\n")

    for i, pdb_id in enumerate(pdb_ids):
        if i % 100 == 0 and i > 0:
            print(f"  Processed {i}/{len(pdb_ids)} | "
                  f"Collected {len(records)} samples | "
                  f"No salt: {no_salt} | Failed: {failed}")

        # Rate limiting — be polite to PDB servers
        if i % 20 == 0:
            time.sleep(0.5)

        # Fetch REMARK 280
        remark = fetch_remark_280(pdb_id)
        if not remark:
            failed += 1
            continue

        # Extract salt concentration
        concentration, salt_type = extract_salt_concentration(remark)
        if concentration is None:
            no_salt += 1
            continue

        # Filter unreasonable values
        if concentration <= 0 or concentration > 5.0:  # > 5M is unrealistic
            continue

        # Fetch sequence
        sequence = fetch_fasta_sequence(pdb_id)
        if not sequence:
            failed += 1
            continue

        records.append({
            "pdb_id": pdb_id,
            "sequence": sequence,
            "seq_length": len(sequence),
            "salt_concentration_M": concentration,
            "salt_type": salt_type,
            "remark_280": remark[:500]  # truncate for storage
        })

        # Stop early if we have enough data
        if len(records) >= 8000:
            print(f"\nReached {len(records)} samples, stopping early.")
            break

    # Save to CSV
    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE")
    print(f"  Total processed: {i+1}")
    print(f"  Valid samples:   {len(records)}")
    print(f"  No salt found:   {no_salt}")
    print(f"  Failed fetches:  {failed}")
    print(f"{'='*60}")

    output_path = Path(output_file)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"\nDataset saved to: {output_path.absolute()}")
    print(f"Columns: {list(records[0].keys())}")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    records = build_dataset(max_search=30000)
    
    # Quick stats
    if records:
        concentrations = [r["salt_concentration_M"] for r in records]
        print(f"\nSalt Concentration Stats:")
        print(f"  Min:    {min(concentrations):.4f} M")
        print(f"  Max:    {max(concentrations):.4f} M")
        print(f"  Mean:   {sum(concentrations)/len(concentrations):.4f} M")
        
        # Salt type distribution
        from collections import Counter
        salt_dist = Counter(r["salt_type"] for r in records)
        print(f"\nSalt Type Distribution (top 10):")
        for salt, count in salt_dist.most_common(10):
            print(f"  {salt}: {count}")
