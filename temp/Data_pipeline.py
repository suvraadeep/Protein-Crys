import os, requests, json, re
import pandas as pd

TEMP_MIN_K = 250
TEMP_MAX_K = 310

def fetch_data(target_n, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = f"{save_dir}/metadata.csv"

    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    all_ids = []
    for start in range(0, target_n * 5, 10000):
        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "exptl_crystal_grow.temp",
                    "operator": "exists"
                }
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": start, "rows": 10000}}
        }

        try:
            r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query",
                              json=query, timeout=60).json()
            ids = [h["identifier"] for h in r.get("result_set", [])]
            all_ids.extend(ids)
            if len(all_ids) >= target_n * 3:
                break
        except:
            continue

    records = []
    for i in range(0, len(all_ids), 100):
        if len(records) >= target_n * 1.5:
            break

        batch = all_ids[i:i+100]
        gql = f"""{{
            entries(entry_ids: {json.dumps(batch)}) {{
                rcsb_id
                exptl_crystal_grow {{ temp }}
                polymer_entities {{
                    entity_poly {{
                        pdbx_seq_one_letter_code_can
                        rcsb_entity_polymer_type
                    }}
                }}
            }}
        }}"""

        try:
            res = requests.post("https://data.rcsb.org/graphql",
                                json={"query": gql}, timeout=60).json()

            for e in res.get("data", {}).get("entries", []):
                if not e.get("exptl_crystal_grow"):
                    continue

                temp = e["exptl_crystal_grow"][0]["temp"]
                if temp is None:
                    continue

                for poly in e.get("polymer_entities", []):
                    ep = poly["entity_poly"]

                    if "protein" not in str(ep.get("rcsb_entity_polymer_type", "")).lower():
                        continue

                    seq = re.sub(r"[^A-Z]", "",
                                 str(ep.get("pdbx_seq_one_letter_code_can", "")).upper())

                    if 40 < len(seq) < 500:
                        records.append({
                            "pdb_id": e["rcsb_id"],
                            "sequence": seq,
                            "temp_k": float(temp)
                        })
                        break
        except:
            continue

    df = pd.DataFrame(records).drop_duplicates(subset=["sequence"]).reset_index(drop=True)
    df = df[(df["temp_k"] >= TEMP_MIN_K) & (df["temp_k"] <= TEMP_MAX_K)]

    if len(df) > target_n:
        df = df.sample(n=target_n, random_state=42)

    df.to_csv(csv_path, index=False)
    return df
