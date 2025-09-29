"""
directions.py - Module de calcul des directions pour RRDH-ED améliorée
Calcul des directions dans le domaine chiffré sans mapping table
"""

import numpy as np
from gmpy2 import mpz, powmod, invert
from math import ceil


def get_M_vector(Nl):
    """
    Génère le vecteur M pour le calcul des directions.
    
    Args:
        Nl: nombre de vertices dans le patch
        
    Returns:
        array: vecteur M avec M(1)=-1 et M(p>1)=1
    """
    M = np.ones(Nl, dtype=int)
    M[0] = -1
    return M


def calculate_F_Nl(Nl, k=4):
    """
    Calcule F(Nl) qui est égale à d_max (la plus grande direction possible pour un Nl donnée). Le F(Nl) de l'article n'est pas utilisé car il y'a des cas où F(Nl) < d_max. 
    
    Args:
        Nl: nombre de vertices dans le patch
        k: facteur de précision utilisé dans le preprocessing
        
    Returns:
        int: valeur maximale des directions non tatouées
    """
    if Nl <= 1:
        return 0
    
    # Formule originale de l'article
    #F_Nl_original = 1.925 * (Nl-1)**3 - 60.6 * (Nl-1)**2 + 528 * (Nl-1) - 609
    
    # Calculer d_max théorique avec k donné
    # d_max = (Nl-1) * max_coord où max_coord = 2*10^k (après preprocessing)
    d_max = (Nl - 1) * (2 * (10 ** k))
    
    # F(Nl) doit être > d_max avec une marge de sécurité
    F_Nl =  2 * d_max

    return F_Nl


def calculate_T_Nl(Nl, t=50):
    """
    Calcule l'intervalle de robustesse T(Nl).
    
    Args:
        Nl: nombre de vertices dans le patch
        t: facteur de robustesse (par défaut 50)
        
    Returns:
        int: taille de l'intervalle de robustesse
    """
    return t * (Nl - 1)


def calculate_B_Nl(Nl, t=50, k=4):
    """
    Calcule le pas de quantification B(Nl).
    
    Args:
        Nl: nombre de vertices dans le patch
        t: facteur de robustesse
        k: facteur de précision utilisé dans le preprocessing
        
    Returns:
        int: pas de quantification
    """
    if Nl <= 1:
        return 0  # Pas de tatouage possible sur un patch d'un seul vertex
    
    F_Nl = calculate_F_Nl(Nl, k)
    T_Nl = calculate_T_Nl(Nl, t)
    B_Nl = ceil((F_Nl + T_Nl) / (Nl - 1))
    return B_Nl





def compute_encrypted_direction(encrypted_patch, j, N):
    """
    Calcule la direction chiffrée Cd pour l'axe j d'un patch.
    
    Args:
        encrypted_patch: patch chiffré [[c_x, c_y, c_z], ...]
        j: axe (0=x, 1=y, 2=z)
        N: module du système Paillier
        
    Returns:
        mpz: direction chiffrée Cd
    """
    N2 = N * N
    Nl = len(encrypted_patch)
    M = get_M_vector(Nl)
    
    # Initialiser le produit
    Cd = mpz(1)
    
    # Calculer le produit selon M
    for i in range(Nl):
        coord_encrypted = encrypted_patch[i][j]
        
        if M[i] == 1:
            # Multiplier par C[i,j]
            Cd = (Cd * coord_encrypted) % N2
        else:  # M[i] == -1
            # Multiplier par C[i,j]^(-(Nl-1))
            C_inv = invert(coord_encrypted, N2)
            C_inv_power = powmod(C_inv, Nl-1, N2)
            Cd = (Cd * C_inv_power) % N2
    
    return Cd

def calculate_direction_from_encrypted(Cd, N, F_limit):
    """
    Calcule la direction d à partir de son chiffré Cd.
    Utilise le fait que g = N+1.
    
    Args:
        Cd: direction chiffrée
        N: module du système Paillier
        F_limit: limite pour déterminer le signe (F(Nl) ou 2F(Nl)+T(Nl))
        
    Returns:
        int: direction en clair
    """
    # Calculer d mod N
    d_mod_N = ((Cd - 1) // N) % N
    
    # Déterminer le signe
    if d_mod_N <= F_limit and d_mod_N >= 0:
        # Direction positive
        d = int(d_mod_N)
    elif d_mod_N >= N - F_limit and d_mod_N <= N-1:
        # Direction négative
        d = int(d_mod_N) - N
    else:
        # Cas imprévu
        d = 0
        print(f"Alerte: direction mod [N] calculée est hors intervalle. Voir dans le fichier directions.py à la fonction calculate_direction_from_encrypted.")
    return d


def compute_all_directions_encrypted(encrypted_patch, N):
    """
    Calcule toutes les 3 directions chiffrées d'un patch chiffré.
    
    Args:
        encrypted_patch: patch chiffré
        N: module du système Paillier
        
    Returns:
        list: [Cd_x, Cd_y, Cd_z]
    """
    directions_encrypted = []
    
    for j in range(3):  # x, y, z
        Cd = compute_encrypted_direction(encrypted_patch, j, N)
        directions_encrypted.append(Cd)
    
    return directions_encrypted


def determine_directions_from_encrypted(directions_encrypted, N, Nl):
    """
    Détermine les directions en clair à partir des directions chiffrées.
    
    Args:
        directions_encrypted: [Cd_x, Cd_y, Cd_z]
        N: module du système Paillier
        Nl: nombre de vertices dans le patch
        
    Returns:
        list: [d_x, d_y, d_z] directions en clair
    """
    F_Nl = calculate_F_Nl(Nl)
    directions = []
    
    for Cd in directions_encrypted:
        d = calculate_direction_from_encrypted(Cd, N, F_Nl)
        directions.append(d)
    
    return directions


def compute_directions_cleartext(patch, j):
    """
    Calcule la direction en clair pour vérification.
    
    Args:
        patch: patch en clair (Nl, 3)
        j: axe (0=x, 1=y, 2=z)
        
    Returns:
        int: direction en clair
    """
    Nl = len(patch)
    M = get_M_vector(Nl)
    
    d = 0
    for i in range(Nl):
        if M[i] == 1:
            d = d + patch[i, j] * M[i]
        else:  # M[i] == -1
            d = d - (Nl-1) * patch[i, j]
            
    return int(d)


def get_watermarking_params(Nl, t=50, k=4):
    """
    Obtient tous les paramètres de tatouage pour un patch.
    
    Args:
        Nl: nombre de vertices dans le patch
        t: facteur de robustesse
        k: facteur de précision utilisé dans le preprocessing
        
    Returns:
        dict: paramètres F(Nl), T(Nl), B(Nl)
    """
    params = {
        'F_Nl': calculate_F_Nl(Nl, k),
        'T_Nl': calculate_T_Nl(Nl, t),
        'B_Nl': calculate_B_Nl(Nl, t, k),
        'Nl': Nl,
        't': t,
        'k': k
    }
    
    return params