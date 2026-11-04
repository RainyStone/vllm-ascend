import sys,json,importlib.metadata as m,importlib.util,hashlib
from pathlib import Path
import ascend_moonep,torch
ws=Path('/mnt/share/x00893896/vllm_ascend_moonep_integration')
sdk=ws/'ascend-moonep'
assert Path(ascend_moonep.__file__).resolve()==sdk/'ascend_moonep/__init__.py'
assert hasattr(torch.classes.AscendMoonEP.Combine(),'compute_weighted')
spec=importlib.util.find_spec('vllm_ascend')
assert Path(spec.origin).resolve()==ws/'vllm-ascend/vllm_ascend/__init__.py'
libs=sorted({l.split()[-1] for l in Path('/proc/self/maps').read_text().splitlines()
             if '.so' in l and ('moonep' in l or 'libshmem' in l)})
assert libs and not any('/precision_fix/' in p for p in libs),libs
d=m.distribution('ascend-moonep')
assert json.loads(d.read_text('direct_url.json'))['url']=='file://'+str(sdk)
record={'host':sys.argv[1],'sdk':ascend_moonep.__file__,'sdk_direct_url':d.read_text('direct_url.json'),
        'vllm_ascend':spec.origin,'compute_weighted':True,'libraries':libs,
        'library_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in libs}}
out=Path(__file__).parent/'results';out.mkdir(exist_ok=True)
(out/('installed_'+sys.argv[1].split('.')[-1]+'.json')).write_text(json.dumps(record,indent=2))
print(json.dumps(record))
