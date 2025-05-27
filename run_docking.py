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
    





