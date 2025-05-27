import os
import json
import subprocess
import requests
from typing import Optional

class AF3Runner:
    """
    A utility class to prepare inputs and run AlphaFold 3 jobs,
    either in bulk (directory-based) or single JSON mode.
    """

    def __init__(self,
                 bulk_script: str = "/home/ubuntu/proj/agent/run_af3_bulk.sh",
                 single_script: str = "/home/ubuntu/proj/agent/run_af3_single.sh"):
        self.bulk_script = bulk_script
        self.single_script = single_script
        self.afdb_base = "https://alphafold.ebi.ac.uk/files"

    def fetch_mmcif_from_afdb(self, uniprot_id: str) -> str:
        """
        Download the mmCIF string for the given UniProt ID from AlphaFold DB.
        """
        url = f"{self.afdb_base}/AF-{uniprot_id}-F1-model_v4.cif"
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.text

    def create_a3m(self, sequence: str, output_path: str) -> None:
        """
        Write a single-sequence A3M file with the provided protein sequence.
        """
        seq = sequence.strip().upper()
        with open(output_path, "w") as f:
            f.write(f">query\n{seq}\n")

    def create_json_input(self,
                          smiles: str,
                          uniprot_id: str,
                          af_input_dir: str,
                          jobname: str,
                          protein_sequence: str) -> str:
        """
        Generate the JSON input file for AlphaFold 3 including mmCIF template,
        unpaired A3M, and ligand SMILES.
        Returns the JSON filepath.
        """
        mmcif = self.fetch_mmcif_from_afdb(uniprot_id)
        a3m_path = os.path.join(af_input_dir, f"{jobname}.a3m")
        self.create_a3m(protein_sequence, a3m_path)
        PROTEIN_INDICES = [i for i in range(1, len(protein_sequence))]
        indices = PROTEIN_INDICES

        payload = {
            "name": jobname,
            "modelSeeds": [1],
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": protein_sequence,
                        "unpairedMsa": a3m_path,
                        "pairedMsa": "",
                        "templates": [
                            {
                                "mmcif": mmcif,
                                "queryIndices": indices,
                                "templateIndices": indices
                            }
                        ]
                    }
                },
                {
                    "ligand": {
                        "id": ["B"],
                        "smiles": smiles
                    }
                }
            ],
            "dialect": "alphafold3",
            "version": 3
        }

        json_path = os.path.join(af_input_dir, f"{jobname}.json")
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=4)
        return json_path

    def prepare_input(self,
                      af_input_dir: str,
                      jobname: str,
                      smiles: str,
                      uniprot_id: str,
                      protein_sequence: str) -> str:
        """
        Create both JSON and A3M input files and verify their existence.
        Returns the JSON filepath.
        """
        json_path = self.create_json_input(smiles, uniprot_id, af_input_dir, jobname, protein_sequence)

        # Verify files
        a3m_path = os.path.join(af_input_dir, f"{jobname}.a3m")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Missing JSON input: {json_path}")
        if not os.path.isfile(a3m_path):
            raise FileNotFoundError(f"Missing A3M input: {a3m_path}")
        return json_path

    def run_bulk(self, input_dir: str) -> None:
        """
        Run the bulk AF3 script against the specified input directory.
        """
        cmd = [self.bulk_script, input_dir]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Bulk AF3 script failed (code {result.returncode})")

    def run_single(self, input_json: str) -> None:
        """
        Run the single-job AF3 script against the specified JSON file.
        """
        cmd = [self.single_script, input_json]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Single AF3 script failed (code {result.returncode})")

    def run_af3(self,
                af_input_dir: Optional[str] = None,
                input_json: Optional[str] = None) -> None:
        """
        Dispatch to either bulk or single AF3 execution.
        Provide either af_input_dir or input_json, but not both.
        """
        if af_input_dir and input_json:
            raise ValueError("Specify only one of af_input_dir or input_json")
        if af_input_dir:
            self.run_bulk(af_input_dir)
        elif input_json:
            self.run_single(input_json)
        else:
            raise ValueError("Must specify af_input_dir or input_json")

# Example usage
if __name__ == "__main__":
    runner = AF3Runner()
    af_input_dir = "/home/ubuntu/proj/agent/af_inputs"
    jobname = "example_job"
    smiles = "CCO"
    uniprot_id = "P12345"
    protein_seq = "MKTWY..."  # actual sequence

    # Prepare inputs
    json_file = runner.prepare_input(af_input_dir, jobname, smiles, uniprot_id, protein_seq)

    # Run single-job AF3
    runner.run_af3(input_json=json_file)

