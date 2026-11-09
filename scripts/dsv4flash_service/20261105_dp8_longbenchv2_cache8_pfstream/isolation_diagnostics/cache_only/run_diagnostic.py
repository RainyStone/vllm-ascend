import json,sys
from pathlib import Path
import run_confirmed_remainder as engine
ROOT=Path(__file__).resolve().parent
SESSION=Path(sys.argv[1])
if __name__=="__main__":
    try:
        engine.run_case('cache_only',"slot")
        (SESSION/"STATUS.json").write_text(json.dumps({"status":"passed","diagnostic_only":True,"requests":128})+"\n")
    except Exception as error:
        (SESSION/"STATUS.json").write_text(json.dumps({"status":"failed","diagnostic_only":True,"error":repr(error)})+"\n")
        raise
