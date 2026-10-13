"""Capture actual current Python environment; run in original model-fitting environment if accessible."""
import json, subprocess, sys, platform, os, datetime
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
out=Path(__file__).resolve().parents[1]
(out/'environment-python.txt').write_text(sys.version+'\n'+platform.platform()+'\n')
with (out/'requirements-lock.txt').open('w') as f:
 subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
mods={}
for m in ['jax','jaxlib','numpyro','arviz','numpy','scipy','pandas','matplotlib']:
 try: mods[m]=version(m)
 except PackageNotFoundError: mods[m]='NOT INSTALLED'
(out/'environment-details.json').write_text(json.dumps({'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'python_executable':sys.executable, 'platform':platform.platform(), 'packages':mods, 'jax_platform_env':os.environ.get('JAX_PLATFORMS')},indent=2))
print('Wrote environment-python.txt, requirements-lock.txt, environment-details.json to',out)
