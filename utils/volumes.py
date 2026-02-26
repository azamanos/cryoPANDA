import os
import time
import subprocess
import numpy as np

CHIMERAX_PATH = os.getenv("CHIMERAX_PATH", "/usr/bin/chimerax")
ALIGN_PATH = 'utils/align.py'

def align_volumes(
    ref_path: str, vol_path: str, apix: float = 1.0, flip: bool = False
) -> None:
    """Align a volume in a .mrc file to another .mrc volume using ChimeraX."""
    vol_path_to_save = f'{ref_path.split(".")[0]}_aligned.map'
    flip_str = "--flip" if flip else ""
    log_file = os.path.splitext(vol_path)[0] + ".txt"
    cmd = f'{CHIMERAX_PATH} --nogui --script "{ALIGN_PATH} {ref_path} {vol_path} '
    cmd += f'{flip_str} -o {vol_path_to_save} -f {log_file} " > {log_file} '

    subprocess.check_call(cmd, shell=True)

def align_volumes_and_report(ref_v_path, v_path):

    align_volumes(ref_v_path, v_path)

    #time.sleep(5)  # Pauses execution for 5 seconds
    report_ = []
    with open(f"{v_path[:-4]}.txt", 'r') as f:
        report = f.readlines()
    for li, line in enumerate(report):
        if line[:15] == '  correlation =':
            eqsplit = line.split('=')
            report_.append([float(eqsplit[1].split(',')[0]), float(eqsplit[2].split(',')[0])])
        if line[:33] == '  Matrix rotation and translation':
            mrtl = li
    rotation, translation = [], []
    for i in range(mrtl+1,mrtl+4):
        rt = report[i].split()
        rotation.append(rt[:3])
        translation.append(rt[3])
    report_ = np.array(report_)
    if not len(report_):
        return 0
    #os.system(f'rm {v_path[:-4]}.txt')
    return report_[:,0].max(), np.array(rotation).astype(float), np.array(translation).astype(float)
