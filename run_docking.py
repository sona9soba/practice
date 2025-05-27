#!/usr/bin/env python3
import argparse
import os
import glob
import gzip
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
from pymol import cmd
"""
from AutoDockTools.MoleculePreparation import AD4ReceptorPreparation, LigandPreparation


class PrankRunner:
    @staticmethod
    def run_p2rank(pdb: str, output_dir: str) -> None:
        prank_exe = "/home/ubuntu/p2rank/prank.sh"
        cmd = [prank_exe, "predict", "-c", "alphafold", "-f", pdb, "-o", output_dir]
        print("Running:", " ".join(cmd))
        os.system(" ".join(cmd))

    @staticmethod
    def extract_grid(pdb_file: str, csv_file: str) -> tuple[tuple[float, float, float], float]:
        df = pd.read_csv(csv_file, header=None)
        df.columns = [
            "name","rank","score","probability","sas_points","surf_atoms",
            "center_x","center_y","center_z","residue_ids","surf_atom_ids"
        ]
        res_nums = [int(x.split("_")[1]) for x in df.loc[0, "residue_ids"].split()]
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("t", pdb_file)
        coords = []
        for model in struct:
            for res in model["A"]:
                if res.get_id()[1] in res_nums:
                    atom = res["CA"] if "CA" in res else next(res.get_atoms())
                    coords.append(atom.get_coord())
            break
        if not coords:
            raise ValueError("No matching residues found")
        arr = np.array(coords)
        centroid = tuple(arr.mean(axis=0))
        span = np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)) * 3
        return centroid, float(span)


class ADTRunner:
    @staticmethod
    def prepare_receptor(pdb: str, pdbqt: str) -> None:
        prep = AD4ReceptorPreparation(
            pdb, removeWater=True, mergeNonPolarH=True,
            addPolarH=True, computeGasteiger=True,
            receptorPDBQT=pdbqt
        )
        prep.writePDBQT()

    @staticmethod
    def prepare_ligand(pdb: str, pdbqt: str) -> None:
        prep = LigandPreparation(
            pdb, mergeNonPolarH=False, addPolarH=True,
            computeGasteiger=True, detectTorsions=True,
            ligandPDBQT=pdbqt
        )
        prep.writePDBQT()

    @staticmethod
    def prepare_gpf(
        pdbqt_file: str, out_gpf: str,
        gridcenter, npts=(126,126,126), spacing=0.375,
        receptor_types=None, ligand_types=None,
        smooth=0.5, dielectric=-0.1465
    ) -> str:
        receptor_types = receptor_types or ['A','C','HD','N','OA','SA']
        ligand_types = ligand_types or ['A','C','HD','N','NA','OA','SA','F','Cl','S','P']
        base = os.path.splitext(os.path.basename(pdbqt_file))[0]
        lines = [
            f"npts {' '.join(map(str,npts))}",
            f"gridfld {base}.maps.fld",
            f"spacing {spacing}",
            f"receptor_types {' '.join(receptor_types)}",
            f"ligand_types {' '.join(ligand_types)}",
            f"receptor {pdbqt_file}"
        ]
        if isinstance(gridcenter, str) and gridcenter.lower()=="auto":
            lines.append("gridcenter auto")
        else:
            lines.append("gridcenter " + " ".join(map(str,gridcenter)))
        lines += [f"smooth {smooth}"]
        for lt in ligand_types:
            lines.append(f"map {base}.{lt}.map")
        lines += [f"elecmap {base}.e.map",
                  f"desolvmap {base}.d.map",
                  f"dielectric {dielectric}"]
        with open(out_gpf,"w") as f:
            f.write("\n".join(lines)+"\n")
        return out_gpf

    @staticmethod
    def run_autogrid4(gpf_file: str) -> None:
        os.system(f"autogrid4 -p {gpf_file}")
        print("AutoGrid4 done:", gpf_file)

    @staticmethod
    def adgpu_cmd(receptor_maps_fld: str, ligand_pdbqt: str) -> str:
        exe = '/home/sona/AutoDock-GPU/bin/autodock_gpu_128wi'
        return f"{exe} -ffile {receptor_maps_fld} --lfile {ligand_pdbqt} --npdb 1 --gbest 1"

    @staticmethod
    def run_cmd() -> None:
        fld = glob.glob('*.maps.fld')[0]
        all_lig = glob.glob('*.pdbqt')
        prev = [f.replace('best_','') for f in glob.glob('best_*')]
        ligs = [l for l in all_lig if l not in prev]
        for lig in ligs:
            os.system(ADTRunner.adgpu_cmd(fld, lig))
            if os.path.exists('best.pdbqt'):
                new = 'best_' + lig
                os.rename('best.pdbqt', new)


    @staticmethod
    def get_best_energy(xml_path: str) -> tuple[str, float]:
        tree = ET.parse(xml_path)
        best = float('inf'); best_id = None
        for run in tree.findall(".//runs/run"):
            eid = run.get("id")
            try:
                val = float(run.findtext("free_NRG_binding").strip())
            except:
                continue
            if val < best:
                best, best_id = val, eid
        if best_id is None:
            raise RuntimeError("No binding energies found")
        return best_id, best

"""
class GNINARunner:
    @staticmethod
    def prepare_gnina(complex_pdb: str) -> tuple[str,str]:
        out = os.path.dirname(complex_pdb)
        base = os.path.splitext(os.path.basename(complex_pdb))[0]
        p_pdb = os.path.join(out, f"{base}_protein.pdb")
        p_sdf = os.path.join(out, f"{base}_ligand.sdf")
        cmd.reinitialize(); cmd.load(complex_pdb,"cplx")
        cmd.select("prot","cplx and polymer.protein"); cmd.save(p_pdb,"prot")
        cmd.select("lig","cplx and not polymer.protein"); cmd.save(p_sdf,"lig")
        cmd.delete("all")
        return p_pdb, p_sdf

    @staticmethod
    def run_cmd(receptor_pdb: str, ligand_sdf: str, output_file: str) -> None:
        os.system(f"gnina --score_only --autobox_ligand {ligand_sdf} "
                  f"-r {receptor_pdb} -l {ligand_sdf} -o {output_file}")

    @staticmethod
    def parse_gnina_output(sdf_gz: str) -> float:
        with gzip.open(sdf_gz, "rt") as fh:
            for line in fh:
                if line.strip()=="> <minimizedAffinity>":
                    return float(next(fh).strip())
        raise RuntimeError("minimizedAffinity not found")
    


cif_dir = "/home/ubuntu/proj/JY/cif"
import glob
runner = GNINARunner()
"""

complex_list = []
for f in glob.glob(f"{cif_dir}/*.cif"):
    complex_list.append(f)
print("Complex list:", complex_list)
for cplx in complex_list:
    runner.prepare_gnina(cplx)

protein_list = []
for p in glob.glob(f"{cif_dir}/*_protein.pdb"):
    protein_list.append(p)
print("Protein list:", protein_list)
ligand_list = [p.replace("_protein.pdb","_ligand.sdf") for p in protein_list]
print("Ligand list:", ligand_list)
assert len(protein_list)==len(ligand_list)
score_dict = {}
for i in range(len(protein_list)):
    protein = protein_list[i]
    ligand = ligand_list[i]
    print("Protein:", protein)
    print("Ligand:", ligand)
    base = os.path.splitext(os.path.basename(protein))[0]
    base = base.lstrip("p9wil5_").rstrip("_model_protein")
    print("Base:", base)
    # run gnina
    out_dir = os.path.dirname(protein)
    out_file = os.path.join(out_dir, f"{base}.sdf.gz")
    # run gnina
    GNINARunner.run_cmd(protein, ligand, out_file)
    # parse the output
    score = GNINARunner.parse_gnina_output(out_file)
    print("Score:", score)
    score_dict[base] = score
print(score_dict)

score_dict = {'2f0e71b5c4f9aca01a9e48cca1158a56517f406a3815ace2bc2979a919a92': -9.07627, 'fcb018894c1c94d8ca4ab6f6baac92c57317133fd6f6a3be842e9d543b46a4ff': -9.00002, '0f43e030ffdf409adfce51576da4f469f68d24d215f82d23c132de9f7e65708b': -5.00671, 'f9a4b8ff155117da5641e06068dc99f3f0aae3bb446e32ed2b2f8198563aa908': -3.84633, '3956a0e813e409078cde37077111d64baebde44be48308d771578a738d47d553': -8.15117, '472f529c7125dffc5f60628dd60babfab6978e033c4a03b9ede04efc5126d77a': -5.77317, '0b17669e4d5a842cb51262d47607ea21527e25bc45a390200e1de6cc9ca82c6': -8.09031, '74779df631bccc40c05f784234a1f89007c9bd1cafe595403e2149d5b6e38d70': -9.12071, 'dfc1802f151f030b43d8a3a4578f9ddc6d11cf92ea69158c8662a0e5099fabfa': -8.75923, '80cf09b3b8afafaa72639ed32d5562529aa8a1a5c50426956d67b0ffa99c4eaf': -9.25261, 'd88dcff5bae3c5d2d7e9e33c05fcb2cbd000aa2fb463cc3a4dfbbe3012942061': -7.42476, '1233701b38294700c41d5d2a48ecb50773be43e35791948149d7e6ba9b900846': -8.15511, '45a0e805130e9a08236082a63d5039d826f19dc1eb963de2ebcdf5676ab8b4c5': -9.24252, 'fa870024ad2ec24d9fee50b83091b9153b9935025d1016ec08a358f3d81f43d4': -8.17385, 'b202ba8387c03370c278101a1fd982336f7dd8aa4c1bc979d588bd0735a86428': -7.70743, '3a0de1861a0a8390b9d5f3dc9815423fe13fe8285c9a68a57a4f60c12a7813f': -8.42994, 'f1eccf6a24061980da37b4fd2b9e5b14922a2cec9783e062c5e7c27763a0e847': -9.08333, 'e8ee3e1e329096d345738720473dbe6af1c6c624ce8081513599a3da785d2095': -8.02747, '7bd3effcea21611b31d74b91a8299acc70c729ff8da8d207fd0ad81a673f5c6c': -7.11877, '8d0ddaf3f808106ea4ab924aeaa99814b823bef9ab1b21f396945e7fd50cfca4': -7.68256, 'cd404bb60e37f2931848d0a8f095f6f449585a118d0bbf8ddcebe8940e5b09d7': -7.20478, 'b782dd20dd221cdfca2f610148044841ab1712c814595440b57b320ace4ac8f7': -8.44739, '6955a8d4ffe4fb901c1458c76d2637c05311d5776709dea065aeed5b524df39': -8.6527, '61bb8a59385b28c405bb99f2a920d1b60673693ca34083af64548a3f167ba5e4': -4.3203, '855ea4cf8cd1b4ccc0979eaa6be41446ae0587be0524144f10b56bcfec58fb4a': -8.76651, 'fdf67d7ba6080d263a36b7d90748f56fc5b80819b67361563d6bd71b0bd3752': -8.64438, '0160759dc74e73b964513fcd37d3ff3b30870dd40665d7d520131cc0a01ec8a5': -8.23362, 'a07eab70942bdc26cdbde96463feaaea08c72c6fa66aca1040ff820de6c334ff': -9.63686, 'afcc3aeb0315b0ecbb0906027c41a2f0e33ab36aab17bea738b72eaffdbf531': -8.56542, 'f5bbfc8de38d1295327333e889f6638f6b1484a5d826ca9d110708bf23a074ea': -9.1867, 'efd312bad658535feccecf0f281d898870b653cfb8a60059bc4a68e7a867ef9': -8.53251, '44c20772624d6894e6d7572d45db8ed9acffea1cd59e12a6f35f533fd902d2bf': -8.28063, 'e8cb9a1e81e9aaea9606f7d78e1bd5551ed0e179de092f41fa8e7ae3975bb1': -9.32222, '1c4d304338b1e36a0946487f49612d8cbea241cd8b86a7b641e85bd7ee0ab694': -7.55875, 'dfa8cc70f6b7603b392936944135e5231b75a352ff24c9d2564d4bdf8908f7e7': -8.33249, 'd13c49445c4b0c32933189fbd10bdc7f263a812504dd9fd66962a5174ef18585': 15.91372, '3437a7f5b1eab26b2022707bac50e95ba01c7ad51fc161bcaeb9f013425403e8': -7.43345, '82b0de5358b19637735231b7e10877b909446e22e11abe68a317d18d1a7fa1d0': -8.55834, 'f4ac6cc5b92095f9be2eeaf83b4ab4d10f2ab5d1736bfdb495125c07742d1584': -6.2392, 'e6e052356caae1b25444a56d23882264bcc9a1c5483086660d5af68698169dba': -8.09376, '86f22b7e8aa575ee5524ff99ded75eef4c464e43e8d9fc2191bd42007b716650': -7.65732, 'dc8ad07a4d76c63002c98008f3f7946b89d37941127e8b42a9da187192434556': -7.01493, 'b5314bb9f260d865408412f4c1a34089d2a9c9853166d6cdc5a1f69087f51c27': -5.32771, 'f6548a2594fc8ef6eb334f4cef28d964c27f691687759b85aa3e2b6641e20d76': -8.72828, 'ef1dfdf7acf291fcc8c130f56e3360c011b02198ae15df24b4fc63f8264f56cf': -9.75356, '04495ec5a0aca2a525f5db63b07401742d33b5a0dde06ea49b64d89b44801dc0': -7.29858, '494227f1b22944c3de0dafc7ba9274d83ef796e6f2d7f45e4f9474c9686ce0a': -4.71848, '82b11034d53647948911f53e0c9284578dbe7bcbaab6f14bdbeec595220637c5': -7.82824, '81d6d5eb29982677c6b3fb3e36d3e1cfdc87a123523cca722ad0944fb997e7': -9.02484, '474320cdb795bbc7c90979d5ab4b6b59d821af587d92faf764c08be66d4c8852': -7.84395, '328d5f5a240d8041bc103fa8112bd5b43f742820f84b5e108c21c182fb5e252c': -9.09294, '449984b69e5734fa7cab10295faba871ac443639e34dfd7b181f2a593acfecc2': -8.51655, '2bf0d38bda70859e3f65f1fd651ef1c9e81ef6ba69fe2ba243990e4164a46238': -4.6069, '2f468fd6bf33cf70312189525a471d28ed1d5cde4ff400b543fae9ea496378': -8.2776, '0c136151f2d3fe1fa8ec5a51c4fb6fdd0bd0abb374a38e6602f2775617741c00': -7.25109, 'f03d698692b0b53f0cb2720186910403ff384903ba89ba05d5fa3f9a9aaab62f': -5.79068, '1ebeec596f5cf07b5f096e335b9a8a1ac613b2e7c2d6ee3e1f31c980426f1e91': -8.46175, '4e95824c325d924cadee18a2c92d8957111453162666f04b09ab6d9ffce50d18': -7.66097, '75457f1feca516aa339a92904f80647fdec913037dd99694d4428b8fc9e53171': -7.80399}
df = pd.DataFrame(list(score_dict.items()), columns=["hash_id", "binding_score"])
print(df)
df.to_csv("ligands_r2_docking.csv", index=False)



import pandas as pd
import requests
import time

# Load CSVs with safe dtype handling
df = pd.read_csv("mce_leadlike.csv", dtype=str)

sele_df = pd.read_csv("ligands_r4_docking_smiles.csv", dtype=str, sep=",", encoding="utf-8")


# Create a hash_id → SMILES mapping
#hash_to_smiles = dict(zip(df["hash_id"], df["smiles"]))

# Add SMILES column to sele_df
smiles_list = []
#missing_hashes = []

#for h in sele_df["hash_id"]:
#    smi = hash_to_smiles.get(h)
#    smiles_list.append(smi)
#    if smi is None:
#        missing_hashes.append(h)

sele_df["smiles"] = smiles_list

# PubChem fetching function
def fetch_pubchem_props(smiles: str):
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/property/" \
          "CID,MolecularWeight,LogP,HBondDonorCount,HBondAcceptorCount,PolarSurfaceArea,RotatableBondCount/JSON"
    try:
        res = requests.get(url, params={"smiles": smiles}, timeout=10)
        res.raise_for_status()
        props = res.json()["PropertyTable"]["Properties"][0]
        return {
            "pubchem_cid": props.get("CID"),
            "mw": props.get("MolecularWeight"),
            "logp": props.get("LogP"),
            "hbd": props.get("HBondDonorCount"),
            "hba": props.get("HBondAcceptorCount"),
            "psa": props.get("PolarSurfaceArea"),
            "rotatable_bonds": props.get("RotatableBondCount")
        }
    except Exception as e:
        return {
            "pubchem_cid": None,
            "mw": None,
            "logp": None,
            "hbd": None,
            "hba": None,
            "psa": None,
            "rotatable_bonds": None
        }

# Fetch and append PubChem properties
pubchem_data = {
    "pubchem_cid": [],
    "mw": [],
    "logp": [],
    "hbd": [],
    "hba": [],
    "psa": [],
    "rotatable_bonds": []
}

print("🔍 Fetching PubChem properties...")
for smi in sele_df["smiles"]:
    if pd.isna(smi):
        for k in pubchem_data:
            pubchem_data[k].append(None)
        continue
    props = fetch_pubchem_props(smi)
    for k in pubchem_data:
        pubchem_data[k].append(props[k])
    time.sleep(0.2)  # polite delay to avoid PubChem rate limiting

# Merge into dataframe
for k in pubchem_data:
    sele_df[k] = pubchem_data[k]

# Save result
sele_df.to_csv("ligands_r4_docking_smiles.csv", index=False)

# Report
#print(f"✅ Matches found: {len(smiles_list) - len(missing_hashes)}/{len(smiles_list)}")
#if missing_hashes:
#    print("⚠️ Missing SMILES for hash_ids:")
#    for h in missing_hashes:
#        print(" >", h)




# that's the base name 
# and save the protein and ligand files to the given directory
# and run gnina 
# and then, with the output, parse the output and save the score to the given csv file, matching the hash 
# search the pubchem about the ligand
# and then, ask biogpt about the ligand, and get the results
# and save the results to the given csv file, matching the hash
# and finally, run plip and get the results, which is saved as the human readable format
# and save the results to the given csv file, matching the hash"

import pandas as pd
import requests
import time
from urllib.parse import quote

# ─── Configuration ─────────────────────────────────────────────────────────────

INPUT_CSV   = "ligands_r4_docking_smiles.csv"
OUTPUT_CSV  = "ligands_r4_docking_pubchem.csv"

HEADERS = {
    "User-Agent": "YourAppName/1.0 (you@example.com)",
    "Accept":     "application/json",
}

OTHER_PROPS = [
    "MolecularWeight",
    "LogP",                 # was XLogP
    "HBondDonorCount",
    "HBondAcceptorCount",
    "PolarSurfaceArea",
    "RotatableBondCount",
]

# ─── Helpers ───────────────────────────────────────────────────────────────────

def get_cid_from_smiles(smiles: str) -> str:
    try:
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
            f"{quote(smiles, safe='')}/cids/TXT"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.text.strip().splitlines()[0]
    except Exception as e:
        print(f"⚠️ CID lookup failed for {smiles!r}: {e}")
        return "NA"


# ─── Main Script ───────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(INPUT_CSV, dtype=str)
    df["pubchem_cid"] = None
    for prop in OTHER_PROPS:
        df[prop] = None

    print("🔎 Fetching CID and properties for each SMILES…")
    for i, smi in df["smiles"].items():
        if pd.isna(smi) or not smi.strip():
            df.at[i, "pubchem_cid"] = "NA"
            continue

        cid = get_cid_from_smiles(smi)
        df.at[i, "pubchem_cid"] = cid
        time.sleep(0.2)
    base = "https://pubchem.ncbi.nlm.nih.gov/compound/"
    df["pubchem_url"] = df["pubchem_cid"].apply(
        lambda cid: f"{base}{cid}" if pd.notna(cid) and cid != "NA" else ""
         )

    # Rename columns to lowercase keys if you prefer:
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Done — saved augmented data to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

"""
import os
import shutil
import pandas as pd

# ─── Configuration ─────────────────────────────────────────────────────────────

CSV_PATH = "ligands_r4_docking_pubchem.csv"
CIF_SRC_DIR = "/home/ubuntu/proj/JY/cif"      # where the hash-named CIFs currently live
CIF_DEST_DIR = "/home/ubuntu/proj/JY/sele"  # where to put the renamed CIFs

# ─── Load the mapping ──────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH, dtype=str)
# Ensure no NaNs
df = df.fillna("")

# Build a dict: hash_id -> pubchem_cid
hash_to_cid = dict(zip(df["hash_id"], df["pubchem_cid"]))

# ─── Process each CIF ──────────────────────────────────────────────────────────

os.makedirs(CIF_DEST_DIR, exist_ok=True)

for hash_id, cid in hash_to_cid.items():
    # skip rows without a valid CID
    if not cid or cid.upper() == "NA":
        print(f"⚠️  Skipping {hash_id}: no valid CID")
        continue

    src_name = f"p9wil5_{hash_id}_model.cif"
    src_path = os.path.join(CIF_SRC_DIR, src_name)

    if not os.path.isfile(src_path):
        print(f"❌  Source file not found: {src_path}")
        continue

    dest_name = f"p9wil5_{cid}.cif"
    dest_path = os.path.join(CIF_DEST_DIR, dest_name)

    try:
        shutil.move(src_path, dest_path)
        print(f"✅  Moved: {src_name} → {dest_name}")
    except Exception as e:
        print(f"❌  Failed to move {src_name}: {e}")
