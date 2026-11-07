"""Regenerate the current study figure set from saved numerical results."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'outputs/displays')
args=parser.parse_args();destination=args.output.resolve()
subprocess.run([sys.executable,'-B',str(ROOT/'additional/src/build_displays.py'),'--output',str(destination)],check=True)
subprocess.run([sys.executable,'-B',str(ROOT/'scripts/make_exploratory_figures.py'),'--output',str(destination)],check=True)
