"""Tests du lot 1 sur des données INVENTÉES. Aucune donnée du match ici.

Lancer depuis la racine du dépôt :
    python tests/test_contact.py
"""

import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.contact import (  # noqa: E402
    LARGEUR, LONGUEUR, TROU_MAX, equipe_brute, lire_tracking,
)

N = 40                         # nombre de frames inventées
EQUIPES = np.array([0, 1])     # piste 0 = domicile (H), piste 1 = extérieur (A)


def scene(ballon_xy, dom_xy, ext_xy):
    """Fabrique (ballon, joueurs) en mètres à partir de trois trajectoires (N, 2)."""
    ballon = np.asarray(ballon_xy, dtype=np.float32)
    joueurs = np.stack([np.asarray(dom_xy, dtype=np.float32),
                        np.asarray(ext_xy, dtype=np.float32)], axis=1)
    return ballon, joueurs


def immobile(x, y):
    return np.tile([x, y], (N, 1)).astype(np.float32)


def test_joueur_seul_au_contact():
    ballon, joueurs = scene(immobile(50, 30), immobile(50.5, 30), immobile(60, 30))
    assert (equipe_brute(ballon, joueurs, EQUIPES) == "H").all()


def test_equipe_exterieure_au_contact():
    ballon, joueurs = scene(immobile(50, 30), immobile(60, 30), immobile(50, 30.6))
    assert (equipe_brute(ballon, joueurs, EQUIPES) == "A").all()


def test_joueur_trop_loin():
    ballon, joueurs = scene(immobile(50, 30), immobile(52, 30), immobile(60, 30))
    assert equipe_brute(ballon, joueurs, EQUIPES).isna().all()


def test_duel_donne_vide():
    ballon, joueurs = scene(immobile(50, 30), immobile(50.5, 30), immobile(50, 30.6))
    assert equipe_brute(ballon, joueurs, EQUIPES).isna().all()


def test_ballon_trop_rapide_donne_vide():
    x = 20 + 0.8 * np.arange(N)  # 0,8 m par frame = 20 m/s : le ballon vole
    ballon = np.column_stack([x, np.full(N, 30)])
    dom = np.column_stack([x, np.full(N, 30.5)])  # un joueur « suit » le ballon à 0,5 m
    ballon, joueurs = scene(ballon, dom, immobile(90, 30))
    assert equipe_brute(ballon, joueurs, EQUIPES).isna().all()


def test_ballon_invisible_donne_vide():
    ballon, joueurs = scene(np.full((N, 2), np.nan), immobile(50.5, 30), immobile(60, 30))
    assert equipe_brute(ballon, joueurs, EQUIPES).isna().all()


def _ballon_lent_avec_trou(debut, longueur):
    x = 40 + 0.1 * np.arange(N)  # 2,5 m/s : un ballon conduit au pied
    vrai = np.column_stack([x, np.full(N, 30)])
    vu = vrai.copy()
    vu[debut:debut + longueur] = np.nan
    dom = np.column_stack([x, np.full(N, 30.5)])  # joueur à 0,5 m de la vraie trajectoire
    return scene(vu, dom, immobile(90, 30))


def test_petit_trou_comble():
    ballon, joueurs = _ballon_lent_avec_trou(10, TROU_MAX)  # trou de 12 frames : comblé
    assert (equipe_brute(ballon, joueurs, EQUIPES) == "H").all()


def test_long_trou_non_comble():
    ballon, joueurs = _ballon_lent_avec_trou(10, TROU_MAX + 1)  # 13 frames : trop long
    serie = equipe_brute(ballon, joueurs, EQUIPES)
    assert serie.iloc[10:10 + TROU_MAX + 1].isna().all()
    assert (serie.iloc[:10] == "H").all() and (serie.iloc[10 + TROU_MAX + 1:] == "H").all()


def test_trou_au_debut_non_comble():
    ballon, joueurs = _ballon_lent_avec_trou(0, 5)  # pas de position connue avant le trou
    serie = equipe_brute(ballon, joueurs, EQUIPES)
    assert serie.iloc[:5].isna().all() and (serie.iloc[5:] == "H").all()


def test_forme_de_la_sortie():
    ballon, joueurs = scene(immobile(50, 30), immobile(50.5, 30), immobile(60, 30))
    serie = equipe_brute(ballon, joueurs, EQUIPES)
    assert len(serie) == N and list(serie.index) == list(range(N))
    assert set(serie.dropna().unique()) <= {"H", "A"}


def test_lire_tracking_format_et_alignement():
    # Trois frames inventées, au format du fichier brut. frame_id 1 = frame vidéo 0.
    lignes = [
        "1:0.5,0.5,1.2;NaN,NaN,NaN:0.5,0.5,3.0\n",
        "2:0.1,0.2,0.0;0.9,0.8,2.0:NaN,NaN,NaN\n",
        "3:NaN,NaN,NaN;0.4,0.6,1.0:0.25,0.75,0.0\n",
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.writelines(lignes)
        chemin = f.name
    try:
        ballon, joueurs = lire_tracking(chemin, n_frames=3, n_pistes=2)
    finally:
        os.remove(chemin)
    assert np.allclose(ballon[0], [0.5 * LONGUEUR, 0.5 * LARGEUR])
    assert np.isnan(ballon[1]).all()
    assert np.allclose(ballon[2], [0.25 * LONGUEUR, 0.75 * LARGEUR])
    assert np.allclose(joueurs[1, 1], [0.9 * LONGUEUR, 0.8 * LARGEUR])
    assert np.isnan(joueurs[0, 1]).all() and np.isnan(joueurs[2, 0]).all()


if __name__ == "__main__":
    tests = [f for nom, f in sorted(globals().items()) if nom.startswith("test_")]
    for test in tests:
        test()
        print("OK  ", test.__name__)
    print("\n{} tests réussis".format(len(tests)))
