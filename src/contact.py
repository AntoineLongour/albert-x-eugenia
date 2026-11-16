"""Lot 1 -- le contact : quelle équipe touche vraiment le ballon, image par image.

Sortie : une pd.Series indexée par frame vidéo (0 à 199 318), qui vaut
    'H'  un joueur de Barcelone (domicile) est au contact du ballon
    'A'  un joueur de l'Athletic (extérieur) est au contact du ballon
    NaN  personne n'est au contact : ballon invisible, libre, en vol, ou duel

La série est « brute » : elle dit qui touche le ballon à cet instant, pas qui a la
possession. C'est le lot 3 (règle des 2 s) qui en tire les possessions.

Chaque paramètre est justifié par le football, aucun n'est réglé sur le corrigé.

Usage, depuis la racine du dépôt :
    python src/contact.py      lit le tracking et affiche un résumé, à garder pour soi
"""

import os
import sys

import numpy as np
import pandas as pd

# Racine du dépôt, pour importer games.py et le code de départ des organisateurs.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES, require, source_frames  # noqa: E402
from starter.baseline import team_of_each_track  # noqa: E402  0 = domicile, 1 = extérieur

# ------------------------------------------------------------------ paramètres
FPS = GAMES[GAME]["fps"]                  # 25 images par seconde
LONGUEUR, LARGEUR = GAMES[GAME]["pitch"]  # 105,3 x 68,0 m : pour passer de 0-1 aux mètres

# Portée d'une jambe tendue : au-delà d'1 m, un joueur ne peut pas jouer le ballon.
R_CONTACT = 1.0  # m

# Vitesse de sprint d'un joueur professionnel, environ 36 km/h. Un ballon plus rapide
# est en vol (passe, tir, dégagement) : personne ne le contrôle, même s'il frôle un joueur.
V_BALLON_MAX = 10.0  # m/s

# Un ballon qui n'est pas touché ne change presque pas de direction en une demi-seconde :
# on comble en ligne droite les trous du ballon d'au plus 0,5 s.
TROU_MAX_S = 0.5  # s
TROU_MAX = int(TROU_MAX_S * FPS)  # 12 frames

DOMICILE, EXTERIEUR = 0, 1


# ------------------------------------------------------------------ lecture
def lire_tracking(raw_path, n_frames, n_pistes):
    """Lit le fichier de tracking brut et renvoie (ballon, joueurs) en mètres.

    ballon  : tableau (n_frames, 2), x et y du ballon, NaN s'il n'est pas vu
    joueurs : tableau (n_frames, n_pistes, 2), x et y de chaque piste, NaN si non vue

    Une ligne du fichier : "<frame_id>:<piste>;<piste>;...:<ballon>", chaque groupe
    valant "x,y,vitesse" en coordonnées 0-1. frame_id commence à 1, la frame vidéo à 0.
    """
    ballon = np.full((n_frames, 2), np.nan, dtype=np.float32)
    joueurs = np.full((n_frames, n_pistes, 2), np.nan, dtype=np.float32)

    with open(raw_path) as fichier:
        for ligne in fichier:
            morceaux = ligne.strip().split(":")
            if len(morceaux) != 3:
                continue
            frame = int(morceaux[0]) - 1
            if not 0 <= frame < n_frames:
                continue

            champs = morceaux[1].replace(";", ",").split(",")
            if len(champs) % 3 == 0:
                pistes = np.array(champs, dtype=np.float32).reshape(-1, 3)[:n_pistes, :2]
            else:  # format inattendu : lecture groupe par groupe, plus lente mais sûre
                pistes = np.array([g.split(",")[:2] for g in morceaux[1].split(";")],
                                  dtype=np.float32)[:n_pistes]
            joueurs[frame, :len(pistes)] = pistes
            ballon[frame] = np.array(morceaux[2].split(",")[:2], dtype=np.float32)

    echelle = np.array([LONGUEUR, LARGEUR], dtype=np.float32)
    ballon *= echelle
    joueurs *= echelle
    return ballon, joueurs


# ------------------------------------------------------------------ calcul
def combler_petits_trous(ballon, trou_max=TROU_MAX):
    """Comble en ligne droite les trous du ballon d'au plus trou_max frames.

    Seuls les trous encadrés par deux positions connues sont comblés. Les longs trous,
    et ceux du début ou de la fin du match, restent NaN.
    """
    ballon = ballon.copy()
    vues = np.flatnonzero(~np.isnan(ballon[:, 0]))
    ecarts = np.diff(vues)
    for k in np.flatnonzero((ecarts > 1) & (ecarts - 1 <= trou_max)):
        a, b = vues[k], vues[k + 1]
        t = np.arange(1, b - a, dtype=np.float32) / (b - a)
        ballon[a + 1:b] = ballon[a] + t[:, None] * (ballon[b] - ballon[a])
    return ballon


def vitesse_ballon(ballon):
    """Vitesse du ballon en m/s, par différence entre la frame d'avant et celle d'après."""
    vx = np.gradient(ballon[:, 0]) * FPS
    vy = np.gradient(ballon[:, 1]) * FPS
    return np.hypot(vx, vy)


def _plus_proche(distances, masque):
    """Distance au joueur le plus proche parmi les pistes du masque, infinie s'il n'y en a pas."""
    if not masque.any():
        return np.full(len(distances), np.inf, dtype=np.float32)
    return distances[:, masque].min(axis=1)


def equipe_brute(ballon, joueurs, equipe_piste):
    """Équipe au contact du ballon à chaque frame : 'H', 'A' ou NaN.

    ballon       : (n, 2) en mètres, NaN si non vu
    joueurs      : (n, S, 2) en mètres, NaN si non vu
    equipe_piste : (S,) 0 si la piste est à domicile (H), 1 si elle est à l'extérieur (A)

    Une équipe est au contact si les trois conditions sont réunies :
      1. un de ses joueurs est à moins de R_CONTACT du ballon ;
      2. aucun adversaire n'est aussi à moins de R_CONTACT (sinon c'est un duel) ;
      3. le ballon va moins vite que V_BALLON_MAX (sinon il est en vol).
    """
    equipe_piste = np.asarray(equipe_piste)
    ballon = combler_petits_trous(ballon)
    vitesse = vitesse_ballon(ballon)

    distances = np.hypot(joueurs[:, :, 0] - ballon[:, None, 0],
                         joueurs[:, :, 1] - ballon[:, None, 1])
    distances = np.where(np.isnan(distances), np.inf, distances)  # non vu = infiniment loin

    d_dom = _plus_proche(distances, equipe_piste == DOMICILE)
    d_ext = _plus_proche(distances, equipe_piste == EXTERIEUR)
    pas_trop_rapide = ~(vitesse > V_BALLON_MAX)  # vitesse inconnue (NaN) : pas de preuve de vol

    contact_dom = (d_dom <= R_CONTACT) & (d_ext > R_CONTACT) & pas_trop_rapide
    contact_ext = (d_ext <= R_CONTACT) & (d_dom > R_CONTACT) & pas_trop_rapide

    labels = np.full(len(ballon), np.nan, dtype=object)
    labels[contact_dom] = "H"
    labels[contact_ext] = "A"
    return pd.Series(labels, name="equipe_brute")


def calculer(data_dir=None, game=GAME):
    """Lit les données du match et renvoie la série equipe_brute complète."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    equipe_piste = team_of_each_track(meta_path)
    ballon, joueurs = lire_tracking(raw_path, source_frames("tactical", game), len(equipe_piste))
    return equipe_brute(ballon, joueurs, equipe_piste)


if __name__ == "__main__":
    serie = calculer()
    print("Résumé à garder pour soi : ne l'envoyez jamais à une IA.")
    print("  frames               {}".format(len(serie)))
    print("  contact domicile (H) {:5.1f} %".format(100 * (serie == "H").mean()))
    print("  contact extérieur (A){:5.1f} %".format(100 * (serie == "A").mean()))
    print("  personne au contact  {:5.1f} %".format(100 * serie.isna().mean()))
