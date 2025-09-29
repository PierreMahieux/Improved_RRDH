"""
encryption.py - Module de chiffrement Paillier pour RRDH-ED améliorée
Chiffrement homomorphe des patches du modèle 3D
"""

from gmpy2 import mpz, powmod, invert
# from paillier import get_paillier_keys, get_r, paillier_encrypt, paillier_decrypt_CRT
from src.utils import paillier
from src.improved_robust_reversible_data_hiding.patch_division import reconstruct_model

def encrypt_vertex(vertex_coords, pub_key, r):
    """
    Chiffre les coordonnées d'un vertex.
    
    Args:
        vertex_coords: array de 3 coordonnées entières positives [x, y, z]
        pub_key: clé publique (N, g)
        r: paramètre aléatoire pour le chiffrement
        
    Returns:
        list: coordonnées chiffrées [cx, cy, cz]
    """
    encrypted_coords = []
    for coord in vertex_coords:
        c = paillier.encrypt_given_r(int(coord), pub_key, r)
        encrypted_coords.append(c)
    
    return encrypted_coords


def encrypt_patch(patch, pub_key, r=None):
    """
    Chiffre un patch complet avec le même r.
    
    Args:
        patch: array numpy (Nl, 3) avec coordonnées entières positives
        pub_key: clé publique (N, g)
        r: paramètre aléatoire (si None, en génère un)
        
    Returns:
        encrypted_patch: liste de coordonnées chiffrées
        r: paramètre r utilisé
    """
    N, g = pub_key
    
    # Générer r si non fourni
    if r is None:
        r = get_r(N)
    
    # Chiffrer chaque vertex du patch
    encrypted_patch = []
    for vertex in patch:
        encrypted_vertex = encrypt_vertex(vertex, pub_key, r)
        encrypted_patch.append(encrypted_vertex)
    
    return encrypted_patch, r


def encrypt_patches(patches, pub_key):
    """
    Chiffre une liste de patches.
    
    Args:
        patches: liste d'arrays numpy représentant les patches
        pub_key: clé publique (N, g)
        
    Returns:
        encrypted_patches: liste de patches chiffrés
        r_values: liste des valeurs r utilisées pour chaque patch
    """
    N, g = pub_key
    encrypted_patches = []
    r_values = []
    
    for patch in patches:
        # Générer un nouveau r pour chaque patch
        r = paillier.generate_r(N)
        encrypted_patch, _ = encrypt_patch(patch, pub_key, r)
        
        encrypted_patches.append(encrypted_patch)
        r_values.append(r)
    
    return encrypted_patches, r_values


def decrypt_vertex(encrypted_coords, priv_key, pub_key):
    """
    Déchiffre les coordonnées d'un vertex.
    
    Args:
        encrypted_coords: liste de 3 coordonnées chiffrées
        priv_key: clé privée
        pub_key: clé publique
        
    Returns:
        list: coordonnées déchiffrées [x, y, z]
    """
    decrypted_coords = []
    for c in encrypted_coords:
        m = paillier.decrypt_CRT(c, priv_key, pub_key)
        decrypted_coords.append(int(m))
    
    return decrypted_coords


def decrypt_patch(encrypted_patch, priv_key, pub_key):
    """
    Déchiffre un patch complet.
    
    Args:
        encrypted_patch: liste de vertices chiffrés
        priv_key: clé privée
        pub_key: clé publique
        
    Returns:
        array: patch déchiffré
    """
    import numpy as np
    
    decrypted_patch = []
    for encrypted_vertex in encrypted_patch:
        decrypted_vertex = decrypt_vertex(encrypted_vertex, priv_key, pub_key)
        decrypted_patch.append(decrypted_vertex)
    
    return np.array(decrypted_patch)


def decrypt_patches(encrypted_patches, priv_key, pub_key):
    """
    Déchiffre une liste de patches.
    
    Args:
        encrypted_patches: liste de patches chiffrés
        priv_key: clé privée
        pub_key: clé publique
        
    Returns:
        list: liste de patches déchiffrés (arrays numpy)
    """
    decrypted_patches = []
    
    for encrypted_patch in encrypted_patches:
        decrypted_patch = decrypt_patch(encrypted_patch, priv_key, pub_key)
        decrypted_patches.append(decrypted_patch)
    
    return decrypted_patches


def generate_keys_for_rrdh(key_size=1024):
    """
    Génère les clés Paillier avec g = N + 1 pour la méthode améliorée.
    
    Args:
        key_size: taille de la clé en bits
        
    Returns:
        pub_key: (N, g) avec g = N + 1
        priv_key: clé privée
    """
    encription_keys = paillier.generate_keys(key_size)
    pub_key= encription_keys["public"]
    priv_key = encription_keys["secret"]
    N, g = pub_key
    
    # Forcer g = N + 1 pour la méthode améliorée
    g = N + 1
    pub_key = (N, g)
    
    return pub_key, priv_key



def get_encryption_info(encrypted_patches, pub_key):
    """
    Obtient des informations sur les patches chiffrés.
    
    Args:
        encrypted_patches: liste de patches chiffrés
        pub_key: clé publique
        
    Returns:
        dict: informations sur le chiffrement
    """
    N, g = pub_key
    
    info = {
        'n_patches': len(encrypted_patches),
        'patches_sizes': [len(patch) for patch in encrypted_patches],
        'key_size': N.bit_length(),
        'g_equals_N_plus_1': (g == N + 1)
    }
    
    return info

def encrypt_isolated_vertices(isolated_coords, pub_key):
    """
    Chiffre les vertices isolés avec des r différents.
    Args:
        isolated_coords: liste d'arrays numpy des coordonnées des vertices isolés
        pub_key: clé publique (N, g)        
    Returns:
        encrypted_isolated_coords: liste des coordonnées chiffrées
    """
    encrypted_isolated_coords = []
    for vertex in isolated_coords:
        r=paillier.generate_r(pub_key[0])
        encrypted_vertex= encrypt_vertex(vertex, pub_key,r)
        encrypted_isolated_coords.append(encrypted_vertex)
    return encrypted_isolated_coords
