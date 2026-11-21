# sextant-meteo

Données météo publiques (vent à 10 m) découpées sur un couloir de course, pour un usage personnel.
- Releases `meteo-AAAAMMJJ` : runs archivés GFS (NOAA), ECMWF IFS open data (CC-BY 4.0), ICON (DWD), au format `.npz`.
- Branche `courant` : derniers runs (`gfs`, `ecmwf`, `icon` en `.bin.gz` + `.json`) et `poids.json`, poids du mélange
  GFS / ECMWF / ICON recalculés après chaque run sur les 3 derniers runs dont le vent réel est connu.
Sources : NOAA NOMADS, ECMWF open data (CC-BY 4.0), Deutscher Wetterdienst open data.
