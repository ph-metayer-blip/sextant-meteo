# Sextant — MET-28 : archivage des modèles météo (vent à 10 m) sur le couloir de course, lancé par GitHub Actions.
# Lecture seule de serveurs publics (GEN-01 : aucun accès au jeu).
#   GFS 0,25°  : NOAA NOMADS, filtre (u, v à 10 m, sous-zone)
#   ECMWF IFS  : open data (archive S3), lecture des seuls messages 10u / 10v par plages d'octets
#   ICON       : DWD open data, grille icosaédrique ramenée sur la grille 0,25° (plus proche voisin)
# Sortie : un fichier .npz par modèle et par run : u, v (int16, centièmes de m/s), lats, lons, echeances (h), run (ISO).
# Usage : python3 archiver.py <dossier_sortie> [--deja liste.txt] [--runs 2]
#   --deja : noms de fichiers déjà archivés (ils ne sont pas retéléchargés) ; --runs : nombre de runs récents essayés par modèle.
import os, sys, io, bz2, json, time, argparse, datetime as dt, urllib.request
import numpy as np, eccodes

NORD, SUD, OUEST, EST = (float(os.environ.get(k, d)) for k, d in (('NORD', 50), ('SUD', 10), ('OUEST', -70), ('EST', 0)))
PAS = 0.25
LATS = np.round(np.arange(NORD, SUD - 1e-9, -PAS), 2)          # nord → sud
LONS = np.round(np.arange(OUEST, EST + 1e-9, PAS), 2)          # ouest → est (degrés −180..180)
ECH = list(range(0, 73, 3)) + list(range(78, 121, 6))          # 0–72 h toutes les 3 h, puis 6 h jusqu'à 120 h
DISPO = {'gfs': 5, 'ecmwf': 9, 'icon': 5}                      # heures après le run où il est complet (prudent)
UA = {'User-Agent': 'Sextant-archivage-meteo (usage personnel, lecture seule)'}

def lire(url, plage=None, essais=4):
    for i in range(essais):
        try:
            h = dict(UA)
            if plage: h['Range'] = 'bytes=%d-%d' % plage
            return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120).read()
        except Exception as e:
            err = e
            if getattr(e, 'code', None) == 404: raise
            time.sleep((15 if getattr(e, 'code', None) in (503, 429) else 5) * (i + 1))
    raise err

def messages(b):
    """Décode les messages GRIB2 d'un tampon : {shortName: (valeurs 2D nord→sud, lat1, lon1, dlat, dlon)}"""
    out = {}; data = b; p = 0
    while True:
        i = data.find(b'GRIB', p)
        if i < 0: break
        n = int.from_bytes(data[i + 8:i + 16], 'big'); g = eccodes.codes_new_from_message(data[i:i + n]); p = i + n
        sn = eccodes.codes_get(g, 'shortName'); nj, ni = eccodes.codes_get(g, 'Nj'), eccodes.codes_get(g, 'Ni')
        v = eccodes.codes_get_values(g).reshape(nj, ni)
        la1 = eccodes.codes_get(g, 'latitudeOfFirstGridPointInDegrees'); lo1 = eccodes.codes_get(g, 'longitudeOfFirstGridPointInDegrees')
        if eccodes.codes_get(g, 'jScansPositively') == 1: v = v[::-1]; la1 = eccodes.codes_get(g, 'latitudeOfLastGridPointInDegrees')
        out[sn] = (v, la1, lo1, eccodes.codes_get(g, 'jDirectionIncrementInDegrees'), eccodes.codes_get(g, 'iDirectionIncrementInDegrees'))
        eccodes.codes_release(g)
    return out

def decouper(v, la1, lo1, dla, dlo):
    """Extrait le couloir d'une grille régulière (lat décroissantes, lon croissantes, éventuellement 0..360)."""
    ii = np.rint((la1 - LATS) / dla).astype(int)
    jj = np.rint(((LONS - lo1) % 360) / dlo).astype(int) % v.shape[1]
    return v[np.ix_(ii, jj)]

# ---------- GFS ----------
def gfs(run):
    d, h = run.strftime('%Y%m%d'), run.strftime('%H')
    def un(e):
        u = (f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?dir=%2Fgfs.{d}%2F{h}%2Fatmos&file=gfs.t{h}z.pgrb2.0p25.f{e:03d}"
             f"&var_UGRD=on&var_VGRD=on&lev_10_m_above_ground=on&subregion=&toplat={NORD}&bottomlat={SUD}&leftlon={OUEST}&rightlon={EST}")
        m = messages(lire(u)); return decouper(*m['10u']), decouper(*m['10v'])
    return parallele(un, 3)

# ---------- ECMWF ----------
# Sources, dans l'ordre : serveur de l'ECMWF (runs des ~4 derniers jours), puis archive S3 (runs plus anciens ; refus
# « SlowDown » fréquents les 16–17/11). Miroir Azure essayé le 17/11 : ne sert pas ces fichiers, retiré.
# S3 seulement sur demande (ECMWF_S3=1, rattrapage) : un run pas encore publié n'y est pas non plus, et ses refus coûtent
# ~6 min de passage (mesure du 17/11).
ECM = ['https://data.ecmwf.int/forecasts'] + (['https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com'] if os.environ.get('ECMWF_S3') == '1' else [])
SOURCES_ECMWF = {}
def ecmwf(run):
    d, h = run.strftime('%Y%m%d'), run.strftime('%H')
    def un(e):
        erreurs = []
        for base in ECM:
            for flux in ('oper', 'scda'):
                nom = f"{base}/{d}/{h}z/ifs/0p25/{flux}/{d}{h}0000-{e}h-{flux}-fc"
                try:
                    idx = lire(nom + '.index', essais=3).decode()
                    pl = [json.loads(l) for l in idx.splitlines() if l.strip()]
                    pl = [l for l in pl if l.get('param') in ('10u', '10v')]
                    if len(pl) < 2: continue
                    res = {}
                    for l in pl: res.update(messages(lire(nom + '.grib2', (l['_offset'], l['_offset'] + l['_length'] - 1), essais=3)))
                    SOURCES_ECMWF[base] = SOURCES_ECMWF.get(base, 0) + 1
                    return decouper(*res['10u']), decouper(*res['10v'])
                except Exception as x:
                    erreurs.append(f"{base.split('/')[2]} {flux}: {getattr(x, 'code', x)}")
        raise RuntimeError(f'ECMWF {d}{h} +{e} h introuvable ({"; ".join(erreurs[-3:])})')
    return parallele(un, int(os.environ.get('PARALLELE_ECMWF', 4)))

# ---------- ICON ----------
CACHE = os.environ.get('CACHE_ICON', os.path.expanduser('~/.cache/sextant_icon'))
def icon_indices(run):
    """Indices des cellules ICON les plus proches de chaque point du couloir (grille R03B07, gardée en cache)."""
    os.makedirs(CACHE, exist_ok=True); f = os.path.join(CACHE, f'idx_{NORD}_{SUD}_{OUEST}_{EST}.npy')
    if os.path.exists(f): return np.load(f)
    from scipy.spatial import cKDTree
    d, h = run.strftime('%Y%m%d'), run.strftime('%H'); c = {}
    for var in ('clat', 'clon'):
        u = f"https://opendata.dwd.de/weather/nwp/icon/grib/{h}/{var}/icon_global_icosahedral_time-invariant_{d}{h}_{var.upper()}.grib2.bz2"
        g = eccodes.codes_new_from_message(bz2.decompress(lire(u))); c[var] = np.radians(eccodes.codes_get_values(g)); eccodes.codes_release(g)
    xyz = lambda la, lo: np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    la, lo = np.meshgrid(np.radians(LATS), np.radians(LONS), indexing='ij')
    _, ix = cKDTree(xyz(c['clat'], c['clon'])).query(xyz(la, lo).reshape(-1, 3))
    ix = ix.reshape(la.shape).astype(np.int32); np.save(f, ix); return ix

def icon(run):
    d, h = run.strftime('%Y%m%d'), run.strftime('%H'); ix = icon_indices(run)
    def un(e):
        r = {}
        for var in ('u_10m', 'v_10m'):
            u = f"https://opendata.dwd.de/weather/nwp/icon/grib/{h}/{var}/icon_global_icosahedral_single-level_{d}{h}_{e:03d}_{var.upper()}.grib2.bz2"
            g = eccodes.codes_new_from_message(bz2.decompress(lire(u))); r[var] = eccodes.codes_get_values(g)[ix]; eccodes.codes_release(g)
        return r['u_10m'], r['v_10m']
    return parallele(un, 8)

def parallele(f, n):
    """Échéances lues en parallèle (n à la fois), rendues dans l'ordre de ECH."""
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(n) as ex: r = list(ex.map(f, ECH))
    return [a for a, _ in r], [b for _, b in r]

MODELES = {'gfs': gfs, 'ecmwf': ecmwf, 'icon': icon}

def runs_recents(mod, n, maintenant):
    r = (maintenant - dt.timedelta(hours=DISPO[mod])).replace(minute=0, second=0, microsecond=0)
    r = r.replace(hour=r.hour - r.hour % 6)
    return [r - dt.timedelta(hours=6 * k) for k in range(n)]

def main():
    a = argparse.ArgumentParser(); a.add_argument('sortie'); a.add_argument('--deja'); a.add_argument('--runs', type=int, default=2)
    a.add_argument('--modeles', default='gfs,ecmwf,icon'); a.add_argument('--run', help='runs précis AAAAMMJJHH, séparés par des virgules (rattrapage)')
    o = a.parse_args(); os.makedirs(o.sortie, exist_ok=True)
    deja = set(open(o.deja).read().split()) if o.deja and os.path.exists(o.deja) else set()
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None); bilan = []
    for mod in o.modeles.split(','):
        runs = [dt.datetime.strptime(x, '%Y%m%d%H') for x in o.run.split(',')] if o.run else runs_recents(mod, o.runs, now)
        for run in runs:
            nom = f'{mod}_{run:%Y%m%d%H}.npz'
            if nom in deja: bilan.append(f'{nom} déjà archivé'); continue
            if mod == 'icon' and not o.run and now - run > dt.timedelta(hours=26): bilan.append(f'{nom} hors de portée (le DWD ne garde que 24 h)'); continue
            t0 = time.time()
            try:
                U, V = MODELES[mod](run)
                np.savez_compressed(os.path.join(o.sortie, nom), u=np.rint(np.array(U) * 100).astype(np.int16), v=np.rint(np.array(V) * 100).astype(np.int16),
                                    lats=LATS, lons=LONS, echeances=np.array(ECH), run=run.isoformat() + 'Z')
                src = (' sources ' + str(SOURCES_ECMWF)) if mod == 'ecmwf' else ''; SOURCES_ECMWF.clear()
                bilan.append(f'{nom} archivé{src} ({os.path.getsize(os.path.join(o.sortie, nom)) // 1024} Ko, {time.time() - t0:.0f} s)')
            except Exception as e:
                bilan.append(f'{nom} ÉCHEC après {time.time() - t0:.0f} s : {e}')
    print('\n'.join(bilan))
    open(os.path.join(o.sortie, 'bilan.txt'), 'w').write('\n'.join(bilan) + '\n')

if __name__ == '__main__':
    main()
