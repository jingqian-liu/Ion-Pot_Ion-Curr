#!/usr/bin/env python3
"""
Download small proteins' structures from AlphaFold DB by filtering UniProt entries by molecular mass (Da).

Example:
  python download_small_afdb.py --max-mass 50000 --max-n 200 --out af_small_pdb --format pdb --reviewed

Requires:
  pip install requests
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from urllib.parse import urlparse, parse_qs

import requests


UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
AFDB_PREDICTION_API = "https://alphafold.ebi.ac.uk/api/prediction/{}"


def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def parse_next_link(link_header: str | None) -> str | None:
    """
    UniProt REST pagination via Link header:
      Link: <https://...>; rel="next"
    """
    if not link_header:
        return None
    for part in [p.strip() for p in link_header.split(",")]:
        if 'rel="next"' in part:
            m = re.search(r"<([^>]+)>", part)
            if m:
                return m.group(1)  # full next URL
    return None



def uniprot_iter_accessions(
    max_mass_da: int,
    reviewed: bool,
    organism_id: str | None,
    max_n: int,
    session: requests.Session,
    page_size: int = 500,
):
    query_terms = [f"mass:[* TO {max_mass_da}]"]
    if reviewed:
        query_terms.append("reviewed:true")
    if organism_id:
        query_terms.append(f"organism_id:{organism_id}")

    query = " AND ".join(query_terms)

    params = {
        "query": query,
        "format": "json",
        "fields": "accession",
        "size": min(max(1, page_size), 500),
    }

    yielded = 0
    next_url = UNIPROT_SEARCH_URL
    first_page = True

    while next_url:
        r = session.get(next_url, params=(params if first_page else None), timeout=60)
        r.raise_for_status()
        data = r.json()

        results = data.get("results", [])
        if not results:
            return

        for item in results:
            acc = item.get("primaryAccession")
            if not acc:
                continue
            yield acc
            yielded += 1
            if yielded >= max_n:
                return

        next_url = parse_next_link(r.headers.get("Link"))
        first_page = False



def afdb_get_file_url(uniprot_acc: str, file_format: str, session: requests.Session) -> str | None:
    """
    Calls AlphaFold DB prediction API to retrieve file URLs robustly
    (avoids hard-coding AF-*-model_v*.pdb patterns).
    """
    url = AFDB_PREDICTION_API.format(uniprot_acc)
    r = session.get(url, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()

    payload = r.json()
    # Historically this endpoint returns a list with one object
    if isinstance(payload, list) and payload:
        obj = payload[0]
    elif isinstance(payload, dict):
        obj = payload
    else:
        return None

    # Try common keys first
    if file_format == "pdb":
        for k in ("pdbUrl", "pdb_url", "pdb"):
            if k in obj and isinstance(obj[k], str) and obj[k].startswith("http"):
                return obj[k]
    else:
        for k in ("cifUrl", "cif_url", "cif", "mmcifUrl", "mmcif_url"):
            if k in obj and isinstance(obj[k], str) and obj[k].startswith("http"):
                return obj[k]

    # Fallback: scan values for something that looks like a pdb/cif URL
    want_ext = ".pdb" if file_format == "pdb" else ".cif"
    for v in obj.values():
        if isinstance(v, str) and v.startswith("http") and want_ext in v:
            return v

    return None


def download_file(url: str, out_path: str, session: requests.Session) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = out_path + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-mass", type=int, default=50000, help="Max molecular mass in Daltons (default: 50000 = 50 kDa)")
    ap.add_argument("--max-n", type=int, default=100, help="Max number of proteins to download")
    ap.add_argument("--out", type=str, default="afdb_small", help="Output directory")
    ap.add_argument("--format", choices=["pdb", "cif"], default="pdb", help="Download format")
    ap.add_argument("--reviewed", action="store_true", help="Only UniProt reviewed (Swiss-Prot) entries")
    ap.add_argument("--organism-id", type=str, default=None, help="NCBI taxonomy id, e.g. 9606 for human")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if output file already exists")
    args = ap.parse_args()

    sess = requests.Session()
    sess.headers.update({"User-Agent": "afdb-small-downloader/1.0 (research)"})

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    print(f"Querying UniProt: mass <= {args.max_mass} Da"
          + (" AND reviewed:true" if args.reviewed else "")
          + (f" AND organism_id:{args.organism_id}" if args.organism_id else "")
          + f" | downloading up to {args.max_n} AlphaFold {args.format.upper()} files")
    print()

    n_ok = 0
    n_no_model = 0
    n_fail = 0

    for acc in uniprot_iter_accessions(
        max_mass_da=args.max_mass,
        reviewed=args.reviewed,
        organism_id=args.organism_id,
        max_n=args.max_n,
        session=sess,
    ):
        out_name = f"{safe_filename(acc)}.{args.format}"
        out_path = os.path.join(out_dir, out_name)

        if args.skip_existing and os.path.exists(out_path):
            print(f"[SKIP] {acc} -> {out_name} (exists)")
            n_ok += 1
            continue

        try:
            file_url = afdb_get_file_url(acc, args.format, sess)
            if not file_url:
                print(f"[NO AFDB MODEL] {acc}")
                n_no_model += 1
                continue

            download_file(file_url, out_path, sess)
            print(f"[OK] {acc} -> {out_name}")
            n_ok += 1

        except Exception as e:
            print(f"[FAIL] {acc}: {e}", file=sys.stderr)
            n_fail += 1

    print()
    print(f"Done. downloaded={n_ok}, no_model={n_no_model}, failed={n_fail}")
    print(f"Output folder: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()

