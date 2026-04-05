import requests
import os
import pandas as pd
import time
import tqdm
import threading
import shutil
from pathlib import Path

def run_foldseek(ensembl_id):
    url = "https://search.foldseek.com/api/ticket"
    matches = sorted([i for i in os.listdir("structures") if i.startswith(ensembl_id)])
    if not matches:
        raise FileNotFoundError(f"No structure input found for {ensembl_id} in structures/")

    # Build an explicit query list instead of relying on matches[0].
    # This avoids missing full-model .cif queries when both files and directories exist.
    paths = []

    # 1) Full-model files (single query per file)
    for item in matches:
        full_path = f"structures/{item}"
        if os.path.isfile(full_path) and item.endswith(".cif"):
            paths.append([full_path, None])

    # 2) Chain directories (one query per chain file)
    for item in matches:
        full_path = f"structures/{item}"
        if os.path.isdir(full_path):
            for chain in sorted(os.listdir(full_path)):
                chain_path = f"{full_path}/{chain}"
                if os.path.isfile(chain_path):
                    paths.append([chain_path, chain.split(".")[0]])

    if not paths:
        raise FileNotFoundError(f"No usable .cif files or chain files found for {ensembl_id}")

    df = pd.DataFrame(columns=["ensembl_id", "chain_id", "ticket_id"])
    for path,chain in paths:
        while True:
            try:
                with open(path, "rb") as f:
                    files = { "q": f }
                    data = [
                        ("mode", "3diaa"),
                        ("database[]", "afdb50"),
                        ("database[]", "afdb-swissprot"),
                        ("database[]", "afdb-proteome"),
                        ("database[]", "pdb100"),
                        ("database[]", "BFVD"),
                        ("database[]", "cath50"),
                        ("database[]", "mgnify_esm30"),
                        ("database[]", "bfmd"),
                        ("database[]", "gmgcl_id"),
                    ]
                    ticket_id = requests.post(url, files=files, data=data).json()["id"]
                    break
            except Exception:
                time.sleep(1)
                continue
        df = pd.concat([df, pd.DataFrame({"ensembl_id": ensembl_id, "chain_id": chain, "ticket_id": ticket_id}, index=[0])], ignore_index=True)
    return df

def status_foldseek(ticket_id):
    url = f"https://search.foldseek.com/api/ticket/{ticket_id}"
    response = requests.get(url).json()
    return response.get("status")

def download_results(ticket_id, ensembl_id, chain_id):
    while True:
        try:
            response = requests.get(f"https://search.foldseek.com/api/result/download/{ticket_id}")
            if chain_id:
                os.makedirs(f"foldseek_results/{ensembl_id}", exist_ok=True)
                Path(f"foldseek_results/{ensembl_id}/{chain_id}.tar.gz").write_bytes(response.content)
            else:
                Path(f"foldseek_results/{ensembl_id}.tar.gz").write_bytes(response.content)
            return True
        except Exception:
            time.sleep(1)
            continue

def cleanup_structures(ensembl_id):
    matches = [i for i in os.listdir("structures") if i.startswith(ensembl_id)]
    for item in matches:
        full_path = Path("structures") / item
        try:
            if full_path.is_dir():
                shutil.rmtree(full_path)
            elif full_path.exists():
                full_path.unlink()
        except Exception:
            # Best effort cleanup; keep pipeline running if filesystem is busy.
            continue

def submit_jobs(ensembl_ids, lock, submission_done):
    global foldseek_check
    with tqdm.tqdm(total=len(ensembl_ids), desc="Submitting Foldseek jobs", position=0, leave=True) as pbar:
        for ensembl_id in ensembl_ids:
            output = run_foldseek(ensembl_id)
            output["status"] = "SUBMITTED"
            output["downloaded"] = False
            with lock:
                foldseek_check = pd.concat([foldseek_check, output], ignore_index=True)
            pbar.update(1)
    submission_done.set()

def download_jobs(ensembl_ids, lock, submission_done):
    global foldseek_check
    downloaded_ensembl = set()
    with tqdm.tqdm(total=len(ensembl_ids), desc="Downloading Foldseek results", position=1, leave=True) as pbar:
        while True:
            with lock:
                snapshot = foldseek_check.copy()

            if snapshot.empty:
                if submission_done.is_set() and len(downloaded_ensembl) >= len(ensembl_ids):
                    break
                time.sleep(1)
                continue

            pending_tickets = snapshot.loc[snapshot["status"] != "COMPLETE", "ticket_id"].tolist()
            for ticket_id in pending_tickets:
                try:
                    status = status_foldseek(ticket_id)
                except Exception:
                    continue
                with lock:
                    mask = (foldseek_check["ticket_id"] == ticket_id) & (foldseek_check["status"] != "COMPLETE")
                    if mask.any():
                        foldseek_check.loc[mask, "status"] = status

            to_download = snapshot.loc[
                (snapshot["status"] == "COMPLETE") & (~snapshot["downloaded"]),
                ["ticket_id", "ensembl_id", "chain_id"],
            ]

            for row in to_download.itertuples(index=False):
                if download_results(row.ticket_id, row.ensembl_id, row.chain_id):
                    should_cleanup = False
                    with lock:
                        foldseek_check.loc[foldseek_check["ticket_id"] == row.ticket_id, "downloaded"] = True
                        ensembl_mask = foldseek_check["ensembl_id"] == row.ensembl_id
                        if foldseek_check.loc[ensembl_mask, "downloaded"].all():
                            if row.ensembl_id not in downloaded_ensembl:
                                downloaded_ensembl.add(row.ensembl_id)
                                should_cleanup = True
                                pbar.update(1)
                    if should_cleanup:
                        cleanup_structures(row.ensembl_id)

            if submission_done.is_set() and len(downloaded_ensembl) >= len(ensembl_ids):
                break

            time.sleep(1)

foldseek_check = pd.DataFrame(columns=["ensembl_id", "chain_id", "ticket_id", "status", "downloaded"])
ensembl_ids = [path.split(".cif")[0] for path in os.listdir("structures")]
os.makedirs("foldseek_results", exist_ok=True)

ensembl_ids = [i for i in ensembl_ids if not (os.path.isfile(f"foldseek_results/{i}.tar.gz") or os.path.isdir(f"foldseek_results/{i}"))]

lock = threading.Lock()
submission_done = threading.Event()

submit_thread = threading.Thread(target=submit_jobs, args=(ensembl_ids, lock, submission_done))
download_thread = threading.Thread(target=download_jobs, args=(ensembl_ids, lock, submission_done))

submit_thread.start()
download_thread.start()

submit_thread.join()
download_thread.join()
