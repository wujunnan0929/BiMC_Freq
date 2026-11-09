"""Print the development Last of the configuration an entry locked: python _sel_last.py select_X.json"""
import json
import sys

rep = json.load(open(sys.argv[1], encoding='utf-8'))
if rep['selected_params'] == 'fallback':
    print(rep['fallback_metrics']['last'])
else:
    print(next(c['last'] for c in rep['candidates'] if c['hash'] == rep['selected_hash']))
