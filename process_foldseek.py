import os
import time
import requests
import pandas as pd

class FoldSeekRunner:

    def __init__(self):
        self.FOLDSEEK_URL = "https://search.foldseek.com/api"
        self.AFDB_URL = "https://alphafold.ebi.ac.uk/files"
        self.email = "kris.rhee.kr@gmail.com"
        self.RCSB_GQL_URL = "https://data.rcsb.org/graphql"


    def search_uniprot_id(self, gene_name: str) -> str | None:
        url = (
            f"https://www.uniprot.org/uniprotkb/search"
            f"?query=gene_exact:{gene_name}&fields=accession&format=txt"
        )
        resp = requests.get(url)
        if resp.ok and resp.text.strip():
            return resp.text.strip().splitlines()[0]
        return None

    def get_fasta_from_uniprot(self, uniprot_id: str) -> str | None:
        url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
        resp = requests.get(url)
        if resp.ok:
            return ''.join(resp.text.splitlines()[1:])
        return None

    def fetch_alphafold_prediction(self, uniprot_id: str) -> str:
        pdb_url = f"{self.AFDB_URL}/AF-{uniprot_id}-F1-model_v1.pdb"
        resp = requests.get(pdb_url)
        resp.raise_for_status()
        outfile = f"afdb_{uniprot_id}.pdb"
        with open(outfile, 'w') as f:
            f.write(resp.text)
        return outfile

    def submit_job_to_foldseek(self, pdb_path: str, email: str) -> dict:
        with open(pdb_path, 'rb') as pdb_file:
            files = {"q": pdb_file}
            data = {
                "email": email,
                "mode": "3diaa",
                "database[]": "pdb100"
            }
            resp = requests.post(f"{self.FOLDSEEK_URL}/ticket?server.dbmanagement=true",
                                 files=files, data=data)
        resp.raise_for_status()
        return resp.json()

    def get_foldseek_result(self, ticket: dict) -> dict:
        ticket_id = ticket["id"]
        status_url = f"{self.FOLDSEEK_URL}/ticket/{ticket_id}"
        while True:
            resp = requests.get(status_url)
            resp.raise_for_status()
            status = resp.json().get("status")
            print("⏳ Foldseek status:", status)
            if status == "COMPLETE":
                break
            if status == "ERROR":
                raise RuntimeError(f"❌ Foldseek job {ticket_id} failed")
            time.sleep(2)
        result_url = f"{self.FOLDSEEK_URL}/result/{ticket_id}/0"
        resp = requests.get(result_url)
        resp.raise_for_status()
        return resp.json()
    def get_smiles_from_pubchem(self, inchi_key: str) -> str:
        """
        Query PubChem by InChIKey to retrieve canonical SMILES.
        """
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchi_key}/property/CanonicalSMILES/TXT"
            resp = requests.get(url, timeout=10)
            if resp.ok and resp.text.strip():
                return resp.text.strip()
        except Exception as e:
            print(f"⚠️ PubChem lookup failed for {inchi_key}: {e}")
        return "NA"


    def parse_foldseek_result(self, result: dict, pident_threshold: float = 0.9) -> dict:
        hits = {}
        try:
            results_block = result.get("results", [])
            if not results_block or "alignments" not in results_block[0]:
                print("⚠️ Foldseek result block is empty or malformed.")
                return hits
            alignment_groups = results_block[0]["alignments"]
            for alignment_list in alignment_groups:
                if not isinstance(alignment_list, list):
                    continue
                for aln in alignment_list:
                    if not isinstance(aln, dict):
                        continue
                    pident = float(aln.get("prob", 0))
                    if pident >= pident_threshold:
                        pdb_id = aln["target"].split("-")[0][:4].upper()
                        tax_name = aln.get("taxName", "Unknown")
                        hits[pdb_id] = tax_name
        except Exception as e:
            print(f"⚠️ Failed to parse Foldseek result: {e}")
        return hits

    def filter_pdb_with_ligands(self, uniprot_id: str, pdb_ids: dict) -> list:
        """
        Query RCSB GraphQL API for ligands >500 Da, then fetch canonical SMILES from PubChem.
        """
        results = []
        for pdb_id, tax in pdb_ids.items():
            try:
                query = {
                    "query": """
                        query structure($id: String!) {
                            entry(entry_id: $id) {
                                nonpolymer_entities {
                                    rcsb_nonpolymer_entity {
                                        pdbx_description
                                    }
                                    nonpolymer_comp {
                                        chem_comp {
                                            name
                                            formula
                                            formula_weight
                                        }
                                        rcsb_chem_comp_descriptor {
                                            InChIKey
                                        }
                                    }
                                }
                            }
                        }
                    """,
                    "variables": {"id": pdb_id}
                }
                resp = requests.post(self.RCSB_GQL_URL, json=query)
                resp.raise_for_status()
                data = resp.json()
                nonpolymers = data.get("data", {}).get("entry", {}).get("nonpolymer_entities", [])

                for ligand in nonpolymers:
                    comp = ligand.get("nonpolymer_comp", {}).get("chem_comp", {})
                    weight = comp.get("formula_weight")
                    if weight is None or weight < 500:
                        continue

                    name = comp.get("name", "Unknown")
                    formula = comp.get("formula", "Unknown")
                    inchi_key = ligand.get("nonpolymer_comp", {}).get("rcsb_chem_comp_descriptor", {}).get("InChIKey")
                    loi_flag = 1 if 'LOI' in name.upper() else 0

                    smiles = self.get_smiles_from_pubchem(inchi_key) if inchi_key else "NA"
                    results.append([uniprot_id, pdb_id, name, formula, weight, inchi_key or "NA", smiles, loi_flag, tax])

            except Exception as e:
                print(f"⚠️ GraphQL or PubChem API error on {pdb_id}: {e}")
        return results

    def run(self, uniprot_ids: list[str], outdir: str):
        import json
        os.makedirs(outdir, exist_ok=True)
        for uid in uniprot_ids:
            try:
                print(f"\n🚀 Processing UniProt ID: {uid}")
                pdb_file = self.fetch_alphafold_prediction(uid)
                print(f"📆 Downloaded AlphaFold structure: {pdb_file}")
                ticket = self.submit_job_to_foldseek(pdb_file, self.email)
                print(f"📨 Submitted Foldseek job: {ticket['id']}")
                result = self.get_foldseek_result(ticket)
                raw_json_path = os.path.join(outdir, f"{uid}_foldseek_raw.json")
                with open(raw_json_path, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"📝 Saved raw Foldseek result to: {raw_json_path}")
                pdb_hits = self.parse_foldseek_result(result)
                print(f"🔍 Found {len(pdb_hits)} PDB hits with prob ≥ 0.9")
                print("PDB hits:", pdb_hits)
                if not pdb_hits:
                    print(f"⚠️ No PDB hits found for {uid} in Foldseek result.")
                    continue
                ligand_data = self.filter_pdb_with_ligands(uid, pdb_hits)
                output_path = os.path.join(outdir, f"{uid}_ligands.csv")
                if ligand_data:
                    cols = ['uniprot_id', 'pdb_id', 'ligand_name', 'ligand_formula', 'molecular_weight',
        'InChIKey', 'smiles', 'LOI', 'taxName']
                    df = pd.DataFrame(ligand_data, columns=cols)
                    df.to_csv(output_path, index=False)
                    print(f"✅ Saved ligand data to: {output_path}")
                else:
                    print(f"⚠️ No ligand info found — CSV not created for {uid}")
            except Exception as e:
                print(f"❌ Error processing {uid}: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Foldseek + RCSB ligand scraper for UniProt IDs")
    parser.add_argument("uniprot_ids", nargs="+", help="One or more UniProt IDs to process")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for .csv and raw Foldseek results")
    args = parser.parse_args()

    runner = FoldSeekRunner()
    runner.run(args.uniprot_ids, args.outdir)
