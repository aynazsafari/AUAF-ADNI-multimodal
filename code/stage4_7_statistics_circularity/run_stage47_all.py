from pathlib import Path
import os, subprocess, sys
workspace=Path(os.environ.get("AUAF_WORKSPACE",Path.cwd())).resolve()
code=workspace/"code"
for script in ["verify_reconstruction.py","stage47_complete_analysis.py","stage47_circularity_retraining.py","finalize_stage47.py"]:
    path=code/script
    print(f"\n=== Running {script} ===",flush=True)
    subprocess.run([sys.executable,str(path)],cwd=workspace,env={**os.environ,"AUAF_WORKSPACE":str(workspace)},check=True)
print("Stage 4.7 pipeline completed.")
