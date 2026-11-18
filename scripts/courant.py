# Sextant — MET-27b : prépare ce que l'extension lit (branche « courant » du dépôt public, servie par raw.githubusercontent.com).
# Pour chaque modèle, le run archivé le plus récent est écrit en binaire compressé : <modele>.bin.gz = int16 petit-boutiste,
# centièmes de m/s, ordre [échéance][u puis v][lat nord → sud][lon ouest → est] ; <modele>.json = run, lats, lons, échéances.
# Usage : python3 courant.py <dossier des .npz> <dossier de sortie> [poids.json]
import sys, os, glob, json, gzip, shutil, numpy as np
D, S = sys.argv[1], sys.argv[2]; os.makedirs(S, exist_ok=True); index = {}
for m in ('gfs', 'ecmwf', 'icon'):
    fs = sorted(glob.glob(f'{D}/{m}_*.npz'))
    if not fs: continue
    z = np.load(fs[-1]); u, v = z['u'], z['v']
    data = np.stack([u, v], axis=1).astype('<i2')            # échéance × composante × lat × lon
    with gzip.open(f'{S}/{m}.bin.gz', 'wb', compresslevel=6) as f: f.write(data.tobytes())
    meta = dict(modele=m, run=str(z['run']), lats=[float(x) for x in (z['lats'][0], z['lats'][-1])], nlat=int(len(z['lats'])),
                lons=[float(x) for x in (z['lons'][0], z['lons'][-1])], nlon=int(len(z['lons'])), pas=0.25,
                echeances=[int(x) for x in z['echeances']], unite='centièmes de m/s', ordre='échéance, (u, v), lat nord→sud, lon ouest→est')
    json.dump(meta, open(f'{S}/{m}.json', 'w'), ensure_ascii=False); index[m] = meta['run']
if len(sys.argv) > 3 and os.path.exists(sys.argv[3]): shutil.copy(sys.argv[3], f'{S}/poids.json'); index['poids'] = json.load(open(sys.argv[3]))['calcule']
json.dump(index, open(f'{S}/index.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(index, ensure_ascii=False))
