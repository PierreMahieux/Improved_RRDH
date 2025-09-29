"""
extraction_restoration_ED.py - Module d'extraction et de restauration dans le domaine chiffré (encryptes domain) 
pour RRDH-ED amélioré Tatouage robuste et réversible par histogram shifting
"""

import numpy as np
from gmpy2 import powmod, invert
from src.improved_robust_reversible_data_hiding.directions import (
    calculate_F_Nl, 
    calculate_T_Nl, 
    calculate_B_Nl,
    compute_all_directions_encrypted,
    calculate_direction_from_encrypted,
    get_M_vector
)

def extract_bit_from_direction(direction, F_Nl, T_Nl):
    """
    Extrait un bit depuis une direction en claire.
    
    Args:
        direction: valeur de la direction en claire
        F_Nl: F(Nl) du patch
        T_Nl: T(Nl) du patch
        
    Returns:
        0 ou 1
    """
    threshold = F_Nl + T_Nl / 2
    
    if abs(direction) > threshold :#supprimer and abs(direction) < 2*F_Nl + T_Nl
        return 1
    else:
        return 0


def extract_watermark_from_patch(watermarked_encrypted_patch, N, k=4):
    """
    Extrait 3 bits d'un patch chiffré tatoué.
    
    Args:
        encrypted_patch: patch chiffré tatoué
        N: module Paillier
        k: facteur de précision utilisé dans le preprocessing
        
    Returns:
        liste de 3 bits extraits
    """
    #watermarked_encrypted_patch=list((watermarked_encrypted_patch))
    Nl = len(watermarked_encrypted_patch)
   
    F_Nl = calculate_F_Nl(Nl,k)
    T_Nl = calculate_T_Nl(Nl)
    
    # Calculer les directions chiffrées tatouées
    directions_watermarked_encrypted = compute_all_directions_encrypted(watermarked_encrypted_patch, N) 
    
    # Limite pour les directions tatouées
    F_limit = 2 * F_Nl + T_Nl
    
    extracted_bits = []
    for Cdw in directions_watermarked_encrypted:
        direction_w = calculate_direction_from_encrypted(Cdw, N, F_limit)
        bit = extract_bit_from_direction(direction_w, F_Nl, T_Nl)
        extracted_bits.append(bit)
    
    return extracted_bits


def extract_watermark_from_model(watermarked_patches, N, expected_length=None, k=4):
    """
    Extrait le watermark complet du modèle.
    
    Args:
        watermarked_patches: patches tatoués
        N: module Paillier
        expected_length: longueur attendue du watermark
        valid_patch_indices: indices des patches qui ont été tatoués
        k: facteur de précision utilisé dans le preprocessing
        
    Returns:
        bits extraits
    """
    all_bits = []
    
    
    # Extraction depuis tous les patches valides (≥2 vertices)
    for patch in watermarked_patches:
        patch_bits = extract_watermark_from_patch(patch, N, k)
        all_bits.extend(patch_bits)
        
            
    # Tronquer si nécessaire
    if expected_length and len(all_bits) > expected_length:
        all_bits = all_bits[:expected_length]
    
    return all_bits

# extracted_bits supprimé dans les arguments de restore_patch pour rendre la fonctions restauré indépendant de l'extraction
def restore_encrypted_patch(watermarked_patch, N, k=4):
    """
    Restaure un patch tatoué.
    
    Args:
        watermarked_patch: patch tatoué
        extracted_bits: bits extraits du patch
        N: module Paillier
        k: facteur de précision utilisé dans le preprocessing
        
    Returns:
        patch restauré (patch chiffré)
    """
    Nl = len(watermarked_patch)
    F_Nl = calculate_F_Nl(Nl,k)
    T_Nl = calculate_T_Nl(Nl)
    B_Nl = calculate_B_Nl(Nl, t=50, k=k)
    

    N2 = N * N
    g = N + 1
    
    # Calculer l'inverse de g^B(Nl)
    g_B = powmod(g, B_Nl, N2)
    theta_g_B = invert(g_B, N2)
    
    # Obtenir les directions
    directions_watermarked_encrypted = compute_all_directions_encrypted(watermarked_patch, N)
    F_limit = 2 * F_Nl + T_Nl
    
    M = get_M_vector(Nl)
    threshold = F_Nl + T_Nl / 2
    
    # Copier le patch pour modification
    watermarked_patch = [vertex[:] for vertex in watermarked_patch]
    # Restaurer chaque direction
    for j in range(3):
        Cdw = directions_watermarked_encrypted[j]
        direction = calculate_direction_from_encrypted(Cdw, N, F_limit)
        if abs(direction) > threshold :# bit = 1 and abs(direction) < 2*F_Nl + T_Nl
            
            if direction >= 0:
                # Restaurer les vertices avec M(p) = 1
                for i in range(Nl):
                    if M[i] == 1:
                        old_val = watermarked_patch[i][j]
                        watermarked_patch[i][j] = (old_val * theta_g_B) % N2
            else:
                # Restaurer le vertex avec M(p) = -1
                for i in range(Nl):
                    if M[i] == -1:
                        old_val = watermarked_patch[i][j]
                        watermarked_patch[i][j] = (old_val * theta_g_B) % N2
                        break
    
    return watermarked_patch


def restore_encrypted_patches_from_watermarking(watermarked_patches, N, k=4):
    """
    Restaure les patches chiffrés tatoués.
    
    Args:
        watermarked_patches: patches chiffrés tatoués
        N: module Paillier
        k: facteur de précision utilisé dans le preprocessing
        
    Returns:
        patches chiffrés
    """
    restored_encrypted_patches = []
    
    for patch in watermarked_patches:
        # Copier le patch
        patch_copy = [vertex[:] for vertex in patch]
        
        # Restaurer
        restored_patch = restore_encrypted_patch(patch_copy, N, k)
        restored_encrypted_patches.append(restored_patch)
    
    return restored_encrypted_patches

def reconstruct_encrypted_model(restored_encrypted_patches, patches_indices, encrypted_isolated_coords, isolated_indices, n_vertices):
    """
    Reconstruit le modèle chiffré complet à partir des patches chiffrés restaurés et vertices chiffrés isolés.
    
    Args:
        restored_encrypted_patches: liste de patches chiffrés restaurés
        patches_indices: indices originaux des vertices dans chaque patch
        encrypted_isolated_coords: coordonnées chiffrées des vertices isolés
        isolated_indices: indices des vertices isolés
        n_vertices: nombre total de vertices dans le modèle
        
    Returns:
        encrypted_vertices: modèle chiffré restauré complet
    """
    # Initialiser le modèle avec des listes vides
    encrypted_vertices = [None] * n_vertices
    
    # Replacer chaque vertex depuis les patches
    for patch, indices in zip(restored_encrypted_patches, patches_indices):
        for vertex_in_patch, original_idx in enumerate(indices):
            encrypted_vertices[original_idx] = patch[vertex_in_patch]
    
    # Replacer les vertices isolés
    if isolated_indices and encrypted_isolated_coords is not None:
        for idx, coords in zip(isolated_indices, encrypted_isolated_coords):
            encrypted_vertices[idx] = coords
    
    return encrypted_vertices




def decrypt_complete_model(encrypted_vertices, priv_key, pub_key):
    """
    Déchiffre le modèle complet.
    
    Args:
        encrypted_vertices: liste de vertices chiffrés
        priv_key: clé privée
        pub_key: clé publique
        
    Returns:
        vertices: array numpy du modèle déchiffré
    """
    from src.improved_robust_reversible_data_hiding.encryption import decrypt_vertex
    import numpy as np
    
    decrypted_vertices = []
    
    for encrypted_vertex in encrypted_vertices:
        if encrypted_vertex is not None:
            decrypted_vertex = decrypt_vertex(encrypted_vertex, priv_key, pub_key)
            decrypted_vertices.append(decrypted_vertex)
    
    return np.array(decrypted_vertices)


