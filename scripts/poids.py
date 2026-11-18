# Sextant — MET-27b : poids du mélange GFS / ECMWF / ICON pour le prochain routage, recalculés après chaque run archivé.
# Pour chaque tranche d'horizon (heures après le run), on prend les 3 derniers runs dont la vérité est déjà connue et on cherche
# les poids (≥ 0, somme 1, pas de 0,1) qui auraient le mieux annoncé le vent du jeu. Vérité au temps V : run GFS de V − 12 h à
# son échéance +12 h (le jeu prend un run à partir de +9 h). Grille : 1° tirée de la grille 0,25° archivée, mer seulement.
# Usage : python3 poids.py <dossier des .npz> <sortie poids.json>
import sys, os, glob, json, datetime as dt, numpy as np
D, SORTIE = sys.argv[1], sys.argv[2]
MOD = ['gfs', 'ecmwf', 'icon']; KN = 1.943844; N_RUNS = int(os.environ.get('N_RUNS', 3))
# Tranches en heures après le run GFS R utilisé par le jeu. Jusqu'à R + 15 h, le jeu garde le run R lui-même (le run suivant
# n'entre qu'à R + 6 + 9 h) : vent du jeu, pas de mélange. H = 12 n'est donc pas mesurable (la vérité serait le run R lui-même).
TRANCHES = [(15, 21, [18]), (21, 39, [24, 30, 36]), (39, 72, [42, 48, 54, 60, 66, 72]), (72, 120, [78, 90, 102, 114])]
C = {}
def lire(m, run):
    k = (m, run)
    if k not in C:
        f = f'{D}/{m}_{run:%Y%m%d%H}.npz'; C[k] = np.load(f) if os.path.exists(f) else None
    return C[k]
def champ(m, run, h):
    z = lire(m, run)
    if z is None: return None
    i = np.where(z['echeances'] == h)[0]
    if not len(i): return None
    return np.stack([z['u'][i[0]], z['v'][i[0]]])[:, ::4, ::4].astype(float) / 100 * KN
def masque_mer(lats, lons):
    la, lo = np.meshgrid(lats, lons, indexing='ij'); m = np.ones(la.shape, bool)
    m &= ~((lo > -10) & (la < 44)); m &= ~((lo > -6) & (la >= 44)); m &= ~((lo < -58) & (la > 42))
    return m
def rms(e, m): return float(np.sqrt((e[0][m] ** 2 + e[1][m] ** 2).mean()))
runs = sorted({dt.datetime.strptime(os.path.basename(f).split('_')[1][:10], '%Y%m%d%H') for f in glob.glob(f'{D}/gfs_*.npz')})
if not runs: sys.exit('aucun run GFS')
z0 = lire('gfs', runs[-1]); MASQUE = masque_mer(z0['lats'][::4], z0['lons'][::4])
maintenant = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
sortie = dict(vent_du_jeu_jusqua_h=15, reference='heures après le run GFS utilisé par le jeu', calcule=maintenant.isoformat(timespec='minutes') + 'Z', methode=f'poids sur les {N_RUNS} derniers runs dont la vérité est connue',
              dernier_run_gfs=runs[-1].isoformat() + 'Z', tranches=[])
def meilleur(cas, mods):
    best = None
    grille = [(i / 10, (10 - i) / 10) for i in range(11)] if len(mods) == 2 else [(i / 10, j / 10, (10 - i - j) / 10) for i in range(11) for j in range(11 - i)]
    for w in grille:
        e = np.mean([rms(c['T'] - sum(wi * c[m] for wi, m in zip(w, mods)), MASQUE) for c in cas])
        if best is None or e < best[0]: best = (e, w)
    return best
def cas_tranche(HS, mods):
    cas = []
    for R in reversed(runs):
        for H in HS:
            V = R + dt.timedelta(hours=H); T = champ('gfs', V - dt.timedelta(hours=12), 12)
            c = {m: champ(m, R, H) for m in mods}
            if T is None or any(x is None for x in c.values()): continue
            c['T'] = T; c['R'] = R; cas.append(c)
        if len({c['R'] for c in cas}) >= N_RUNS: break
    return cas
for de, a, HS in TRANCHES:
    # d'abord les 3 modèles ; s'il manque l'historique d'ICON (le DWD ne garde que 24 h), GFS et ECMWF seuls, signalé
    for mods in (MOD, ['gfs', 'ecmwf']):
        cas = cas_tranche(HS, mods); runs_utilises = sorted({c['R'] for c in cas})
        if len(runs_utilises) >= N_RUNS: break
    if len(runs_utilises) < N_RUNS:
        sortie['tranches'].append(dict(de=de, a=a, poids=None, raison=f'{len(runs_utilises)} run(s) complet(s) seulement')); continue
    e, w = meilleur(cas, mods); poids = {m: 0.0 for m in MOD}; poids.update(dict(zip(mods, w)))
    err = {m: round(float(np.mean([rms(c['T'] - c[m], MASQUE) for c in cas])), 2) for m in mods}
    err['moyenne'] = round(float(np.mean([rms(c['T'] - sum(c[m] for m in mods) / len(mods), MASQUE) for c in cas])), 2)
    sortie['tranches'].append(dict(de=de, a=a, poids=poids, modeles=mods, erreur_melange=round(e, 2), erreurs=err,
                                   runs=[r.isoformat() + 'Z' for r in runs_utilises], echeances=len(cas),
                                   **({'note': 'ICON sans historique suffisant : GFS et ECMWF seuls'} if len(mods) == 2 else {})))
json.dump(sortie, open(SORTIE, 'w'), indent=1, ensure_ascii=False)
print(json.dumps(sortie, indent=1, ensure_ascii=False))
