import subprocess
import requests
import re
import os, glob,shutil
from pymol import cmd
import contextlib
import numpy as np
import pandas as pd
import tqdm
import warnings
import time
from ciliary_genes import load_ciliary_ensembl_ids
warnings.filterwarnings("ignore")

cilia_carta = pd.read_csv('CiliaCarta.csv')


ensembl_ids = load_ciliary_ensembl_ids(cilia_carta)

def reviewed_status(uniprot_id):
    while True:
        try:
            resp = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json")
            data = resp.json()
        except Exception:
            return False
        return "uniprotkb reviewed (swiss-prot)" in data.get("entryType", "").lower()


def get_uniprot_ids(ensembl_id):
    try:
        r = requests.get(
            f"https://rest.ensembl.org/xrefs/id/{ensembl_id}",
            headers={"Content-Type": "application/json","User-Agent":"foldseek-script/1.0"},
            params={"content-type": "application/json"},
            timeout=30,
        )
        data = r.json()
    except Exception:
        structs = []
        total_length = canonical_sequence_length(uniprot_id)
        try:
            resp = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json")
            data = resp.json()
        except Exception:
            return structs
        for x in data.get("uniProtKBCrossReferences", []):
            if x.get("database", "").upper() in ["PDB"]:
                props = {p.get("key"): p.get("value") for p in x.get("properties", []) or []}
                for part in props.get("Chains", "").split(", "):
                    chains,range_ = part.split("=")
                    start,end = range_.split("-")
                    coverage = (int(end)-int(start)+1)/total_length
                    resolution = props.get("Resolution")
                    if resolution is None or resolution == "-":
                        res_value = None
                    else:
                        try:
                            res_value = float(resolution.split(" ")[0])
                        except Exception:
                            res_value = None
                    structs.append({
                        "model_id": x.get("id"),
                        "chains": chains.split("/"),
                        "confidence": 1,
                        "coverage": coverage,
                        "resolution": res_value,
                        "quality-score": coverage*0.7 + 0.3, 
                        "start": int(start),
                        "end": int(end),
                        "structure_url": None,
                        "sequence": None,
                    })
        return structs
</Query>'''
    r = requests.post("https://www.ensembl.org/biomart/martservice",
                     data={"query": q},
                     headers={"User-Agent":"foldseek-script/1.0"},
    r.raise_for_status()
    ids = [u for line in r.text.strip().splitlines()
             for u in line.split("\t") if u]
    for uid in ids:
        if reviewed_status(uid):
            return [uid]
    return ids

def isoform_numbers(uniprot_id):
    resp = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.txt", headers={"User-Agent": "foldseek-script/1.0"})
    matches = re.findall(r"IsoId=([A-Za-z0-9\-]+)", resp.text)
    if "-" not in uniprot_id:
        return "1"
    return matches[0].split("-")[1]

def canonical_sequence_length(uniprot_id):
    while True:
        try:
            resp = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json", headers={"User-Agent": "foldseek-script/1.0"})
            data = resp.json()
            length = data.get("sequence", {}).get("length")
            return int(length)
        except Exception:
            time.sleep(1)
            continue

def swiss_structures(uniprot_id):
    structs = []
    try:
        resp = requests.get(f"https://swissmodel.expasy.org/repository/uniprot/{uniprot_id}.json")
        data = resp.json()
    except Exception:
        return structs
    try:
        if data.get("result").get("uniprot_entries")[0].get("isoid") == int(isoform_numbers(uniprot_id)):
            for structure in data.get("result").get("structures"):
                try:
                    structs.append({
                        "model_id": structure.get("template"),
                        "chains": None,
                        "confidence": float(structure.get("qmean").get("avg_local_score")),
                        "coverage": float(structure.get("coverage")),
                        "resolution": None,
                        'quality-score': float(structure.get("qmean").get("avg_local_score"))*0.3 + float(structure.get("coverage"))*0.7,
                        "start": int(structure.get("chains")[0].get("segments")[0].get("smtl").get("from")),
                        "end": int(structure.get("chains")[0].get("segments")[0].get("smtl").get("to")),
                        "structure_url": structure.get("modelcif") if structure.get("modelcif") else structure.get("coordinates"),
                        "sequence": structure.get("chains")[0].get("segments")[0].get("smtl").get("aligned_sequence"),
                    })
                except Exception:
                    continue
    except Exception:
        pass
    return structs

def alphafold_structures(uniprot_id):
    structs = []
    try:
        resp = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}")
        data = resp.json()
    except Exception:
        return structs
    for x in data:
        if "-" not in x.get("uniprotAccession"):
            structs.append({
                "model_id": x.get("modelEntityId"),
                "chains": None,
                "confidence": float(x.get("globalMetricValue"))/100,
                "coverage": float((x.get("sequenceEnd") - x.get("sequenceStart") + 1)/(x.get("uniprotEnd") - x.get("uniprotStart") + 1)),
                "resolution": None,
                "quality-score": float(x.get("globalMetricValue"))*0.003 + float((x.get("sequenceEnd") - x.get("sequenceStart") + 1)/(x.get("uniprotEnd") - x.get("uniprotStart") + 1))*0.7,
                "start": int(x.get("sequenceStart")),
                "end": int(x.get("sequenceEnd")),
                "structure_url": x.get("cifUrl"),
                "sequence": x.get("sequence"),
            })
    return structs

def fethn_chains(model_id, chains,ensembl_id):
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            if model_id not in cmd.get_names():
                cmd.fetch(model_id)
    sequences = {}
    os.makedirs(f"structures/{ensembl_id}", exist_ok=True)
    for i, chain in enumerate(chains):
        cmd.create(f"{model_id}_{i}", f"{model_id} and chain {chain}")
        sequences[chain] = cmd.get_fastastr(f"{model_id}_{i}").split("\n", 1)[1].replace("\n", "")
        cmd.save(f"structures/{ensembl_id}/{chain}.cif", f"{model_id}_{i}")
    for path in glob.glob("*.cif"):
        os.remove(path)
    return sequences


def uniprot_structures(uniprot_id):
    structs = []
    total_length = canonical_sequence_length(uniprot_id)
    for x in requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json").json().get("uniProtKBCrossReferences"):
        if x.get("database").upper() in ["PDB"]:
            props = {p.get("key"): p.get("value") for p in x.get("properties", []) or []}
            for part in props.get("Chains", "").split(", "):
                chains,range_ = part.split("=")
                start,end = range_.split("-")
                coverage = (int(end)-int(start)+1)/total_length
                resolution = props.get("Resolution")
                if resolution is None or resolution == "-":
                    res_value = None
                else:
                    try:
                        res_value = float(resolution.split(" ")[0])
                    except Exception:
                        res_value = None
                structs.append({
                    "model_id": x.get("id"),
                    "chains": chains.split("/"),
                    "confidence": 1,
                    "coverage": coverage,
                    "resolution": res_value,
                    "quality-score": coverage*0.7 + 0.3, 
                    "start": int(start),
                    "end": int(end),
                    "structure_url": None,
                    "sequence": None,
                })
    return structs

columns = ["ensembl_id", "gene_name", "uniprot_id", "model_id", "chains", "confidence", "coverage", "resolution", "start", "end", "sequence", "source", "reviewed", "structure_url"]


os.makedirs("structures", exist_ok=True)
done = [i.split(".")[0] for i in os.listdir("structures")]
ensembl_ids = [i for i in ensembl_ids if i not in done]

final_df = pd.read_csv("ciliary_structures.csv", index_col=False) if os.path.exists("ciliary_structures.csv") else pd.DataFrame(columns=columns)
skipped = pd.read_csv("skipped_genes.csv", index_col=False) if os.path.exists("skipped_genes.csv") else pd.DataFrame(columns=["ensembl_id"])
ensembl_ids = [i for i in ensembl_ids if i not in skipped["ensembl_id"].values]
cilia_carta = pd.read_csv("CiliaCarta.csv")
with tqdm.tqdm(total=len(ensembl_ids), desc="Processing genes") as pbar:
    for ensembl_id in ensembl_ids:
        uniprot_ids = get_uniprot_ids(ensembl_id)
        df = pd.DataFrame(columns=columns)
        gene_name = cilia_carta[cilia_carta["Ensembl Gene ID"] == ensembl_id]["Associated Gene Name"].iloc[0]
        for uniprot_id in uniprot_ids:
            df = pd.concat([df, pd.DataFrame(uniprot_structures(uniprot_id)).assign(source="pdb", ensembl_id=ensembl_id, gene_name=gene_name, uniprot_id=uniprot_id)], ignore_index=True)
            df = pd.concat([df, pd.DataFrame(swiss_structures(uniprot_id)).assign(source="swissmodel", ensembl_id=ensembl_id, gene_name=gene_name, uniprot_id=uniprot_id)], ignore_index=True)
            df = pd.concat([df, pd.DataFrame(alphafold_structures(uniprot_id)).assign(source="alphafold", ensembl_id=ensembl_id, gene_name=gene_name, uniprot_id=uniprot_id)], ignore_index=True)
            if not df.empty:
                df.loc[df["uniprot_id"]==uniprot_id, "isoform"] = f"{uniprot_id}-{isoform_numbers(uniprot_id)})"
        if not df.empty:
            df = df[df["quality-score"] == df["quality-score"].max()]
            df['source'] = pd.Categorical(df['source'], categories=['pdb', 'alphafold', 'swissmodel'], ordered=True)
            if df["source"].iloc[0] == "pdb":
                row = df.sort_values("resolution").iloc[0]
            else:    
                row = df.sort_values("source").iloc[0]
            row["reviewed"] = reviewed_status(row["uniprot_id"])
            if row["source"] == "pdb":
                sequences = fethn_chains(row["model_id"], row["chains"],ensembl_id)
                total_length = canonical_sequence_length(row["uniprot_id"])
                max_index = int(np.argmax([len(sequence)/total_length for sequence in sequences]))
                row["sequence"] = sequences[row["chains"][max_index]]
            else:
                outpath = f"structures/{row['ensembl_id']}.cif"
                subprocess.run(["wget", "-q", "-O", outpath, row["structure_url"]], check=True)
            final_df = pd.concat([final_df, row.to_frame().T], ignore_index=True)
            shutil.rmtree("pdb", ignore_errors=True)
            final_df.drop(columns=["structure_url"], inplace=True)
            final_df.to_csv("ciliary_structures.csv", index=False)
        else:
            skipped = pd.concat([skipped, pd.DataFrame({"ensembl_id": ensembl_id}, index=[0])], ignore_index=True)
            skipped.to_csv("skipped_genes.csv", index=False)
        pbar.update(1)
        
