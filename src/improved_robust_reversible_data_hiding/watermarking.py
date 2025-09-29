"""
watermarking.py - Module de tatouage pour RRDH-ED améliorée
Tatouage robuste et réversible par histogram shifting
"""

import numpy as np
from gmpy2 import mpz, powmod, invert
from src.improved_robust_reversible_data_hiding.directions import (
    calculate_F_Nl, 
    calculate_T_Nl, 
    calculate_B_Nl,
    compute_all_directions_encrypted,
    determine_directions_from_encrypted,
    get_M_vector,
)


def embed_bit_in_patch(encrypted_patch, bit, j, direction, params, N):
    """
    Tatoue un bit dans une direction d'un patch.
    
    Args:
        encrypted_patch: patch chiffré à tatouer
        bit: 0 ou 1 à tatouer
        j: axe (0=x, 1=y, 2=z)
        direction: direction de cet axe
        params: paramètres F(Nl), T(Nl), B(Nl)
        N: module Paillier
        
    Returns:
        patch tatoué (modifié en place)
    """
    if bit == 0:
        # Pas de modification pour le bit 0
        return encrypted_patch
    
    # Déterminer quelles coordonnées modifier
    encrypted_patch = encrypted_patch.copy()  # Pour éviter de modifier l'original
    Nl = len(encrypted_patch)
    
    
    # Bit 1: décaler les coordonnées
    F_Nl = params['F_Nl']
    B_Nl = params['B_Nl']
    N2 = N * N
    
    # Calculer g^B(Nl) mod N² 
    g = N + 1
    g_B = powmod(g, B_Nl, N2)
    
    M =get_M_vector(Nl)
    
    if 0 <= direction <= F_Nl:
        # Direction positive: modifier les vertices avec M(p) = 1
        for i in range(Nl):
            if M[i] == 1:
                old_val = encrypted_patch[i][j]
                encrypted_patch[i][j] = (old_val * g_B) % N2
                
    elif -F_Nl <= direction < 0:
        # Direction négative: modifier le vertex avec M(p) = -1
        for i in range(Nl):
            if M[i] == -1:
                old_val = encrypted_patch[i][j]
                encrypted_patch[i][j] = (old_val * g_B) % N2
                break

    return encrypted_patch


def embed_watermark_in_patch(encrypted_patch, watermark_bits, N, k=4):
    """
    Tatoue 3 bits dans un patch (1 bit par direction).
    
    Args:
        encrypted_patch: patch chiffré
        watermark_bits: liste de 3 bits [bx, by, bz]
        N: module Paillier
        k: facteur de précision utilisé dans le preprocessing
        
    Returns:
        patch tatoué
    """
    Nl = len(encrypted_patch)
    params = {
        'F_Nl': calculate_F_Nl(Nl, k),
        'T_Nl': calculate_T_Nl(Nl),
        'B_Nl': calculate_B_Nl(Nl, t=50, k=k)
    }
    
    # Calculer les directions originales
    directions_encrypted = compute_all_directions_encrypted(encrypted_patch, N)
    directions = determine_directions_from_encrypted(directions_encrypted, N, Nl)
    
    # Tatouer chaque direction
    watermarked_encrypted_patch = encrypted_patch
    for j in range(3):
        if j < len(watermark_bits):
            bit = watermark_bits[j]
            direction = directions[j]
            watermarked_encrypted_patch = embed_bit_in_patch(watermarked_encrypted_patch, bit, j, direction, params, N)
    
    return watermarked_encrypted_patch


def embed_watermark_in_model(encrypted_patches, watermark_bits, N, k=4):
    """
    Tatoue le watermark dans tous les patches du modèle.
    
    Args:
        encrypted_patches: liste de patches chiffrés
        watermark_bits: bits du watermark
        N: module Paillier
        k: facteur de précision utilisé dans le preprocessing
        
    Returns:
        patches tatoués, nombre de bits tatoués
    """
    watermarked_patches = []
    nb_watermarked_bits = 0
    #valid_patch_indices = []
    
    if len(watermark_bits) < 3 * len(encrypted_patches):        
        print(f"Attention: watermark trop court ({len(watermark_bits)} bits) pour {len(encrypted_patches)} patches")
        watermark_bits = watermark_bits + [0] * (3 * len(encrypted_patches) - len(watermark_bits))
        print(f"  -> complété à {len(watermark_bits)} bits avec des 0")

    
    for i, patch in enumerate(encrypted_patches):
        # Copier le patch
        patch_copy = [vertex[:] for vertex in patch]
        patch_copy = embed_watermark_in_patch(patch_copy, watermark_bits[3*i:3*i+3], N, k)
        watermarked_patches.append(patch_copy)
        nb_watermarked_bits += 3


    # patches = encrypted_patches.copy()
    # for w_i in range(len(watermark_bits)):
    #     if watermark_bits[w_i] == 0:
    #         continue

    #     patch = patches[w_i//3]
    #     j = w_i%3


    #     Nl = len(patch)
    #     params = {
    #         'F_Nl': calculate_F_Nl(Nl, k),
    #         'T_Nl': calculate_T_Nl(Nl),
    #         'B_Nl': calculate_B_Nl(Nl, t=50, k=k)
    #     }
    #     # Calculer les directions originales
    #     directions_encrypted = compute_all_directions_encrypted(patch, N)
    #     directions = determine_directions_from_encrypted(directions_encrypted, N, Nl)
    #     bit = watermark_bits[w_i]
    #     direction = directions[j]
    #     patch = embed_bit_in_patch(patch, bit, j, direction, params, N)


    return watermarked_patches, nb_watermarked_bits



def calculate_ber(original, extracted):
    """Calcule le Bit Error Rate"""
    errors = sum(a != b for a, b in zip(original, extracted))
    return errors / len(original)