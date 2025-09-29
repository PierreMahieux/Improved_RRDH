import os
from gmpy2 import mpz, powmod, invert, gcd, mpz_random, random_state, mpz_urandomb, next_prime
import time
import datetime, hashlib
import numpy as np

from math import ceil

from src.utils import paillier, util, mesh_utils

def run(config: dict, encryption_keys:dict, watermarks: tuple, model):
    print("Run RRDH")
    vertices = model["vertices"]
    faces = model["faces"]
    n_vertices = len(vertices)
    pub_key = encryption_keys["public"]
    priv_key = encryption_keys["secret"]
    N, g = pub_key

    watermark = watermarks[0]

    result = {"config": config}

    # Preprocessing
    print("Pre-processing")
    vertices_prep, prep_info = preprocess_vertices(vertices, config["quantisation_factor"])
    # mesh_utils.save_3d_model(vertices_prep, faces, os.path.join(config["result_folder"],"preprocessed.obj"))

    #Patch division
    (patches, patch_indices), (isolated_coords, isolated_indices) = divide_into_patches(vertices_prep.copy(), faces)
    patch_info = get_patch_info(patches, isolated_coords)

    result["patch_number"] = len(patch_indices)
    result["max_vertices_per_patch"] = max([len(patch_indices[i]) for i in range(len(patch_indices))])
    result["min_vertices_per_patch"] = min([len(patch_indices[i]) for i in range(len(patch_indices))])
    
    # Encryption
    print("Encryption")
    start_encryption = time.time()
    encrypted_vertices = encrypt_vertices(vertices_prep, pub_key)
    result["time_encryption"] = time.time() - start_encryption
    
    # Embedding  
    print("Embedding")
    start_embedding = time.time()
    watermarked_encrypted_vertices = embed(encrypted_vertices.copy(), patch_indices, watermark, pub_key, config)
    result["time_embedding"] = time.time() - start_embedding


    # Restauration et sauvegarde du modèle tatoué déchiffré
    watermarked_decrypted_vertices = watermarked_encrypted_vertices.copy()
    for v_i in range(len(watermarked_decrypted_vertices)):
        for c_i in range(3):
            watermarked_decrypted_vertices[v_i][c_i] = int(paillier.decrypt_CRT(watermarked_encrypted_vertices[v_i][c_i], priv_key, pub_key))
    watermarked_restored = inverse_preprocess_vertices(watermarked_decrypted_vertices, prep_info)
    # mesh_utils.save_3d_model(watermarked_restored, faces, os.path.join(config["result_folder"],"watermarked_restored.obj"))

    # Extraction
    print("Extraction")
    start_extraction = time.time()
    (watermarked_patches, watermarked_patch_indices), (isolated_coords, isolated_indices) = divide_into_patches(watermarked_encrypted_vertices, faces)

    extracted_watermark = extract(watermarked_encrypted_vertices, watermarked_patch_indices, encryption_keys, config)
    result["time_extraction"] = time.time() - start_extraction

    # Restoration
    print("Restoration")
    restored_encrypted_vertices = restore_encrypted_vertices(watermarked_encrypted_vertices.copy(), extracted_watermark, watermarked_patch_indices, encryption_keys["public"], config)
    
    restored_decrypted_vertices = restored_encrypted_vertices.copy()
    for v_i in range(len(restored_decrypted_vertices)):
        for c_i in range(3):
            restored_decrypted_vertices[v_i][c_i] = int(paillier.decrypt_CRT(restored_encrypted_vertices[v_i][c_i], priv_key, pub_key))
    restored_decrypted_vertices = inverse_preprocess_vertices(restored_decrypted_vertices, prep_info)

    # mesh_utils.save_3d_model(restored_decrypted_vertices, faces, os.path.join(config["result_folder"],"fully_restored.obj"))

    result["BER"] = util.compare_bits(extracted_watermark, watermark)
    
    return result


def encrypt_vertices(vertices, public_key) -> np.array:
    encrypted_vertices = []
    for v in range(len(vertices)):
        enc = []
        for coord in range(3):
            enc.append(paillier.encrypt_given_r(int(vertices[v][coord]), public_key, 1))
        encrypted_vertices.append(enc)
    return np.array(encrypted_vertices)

def embed(vertices, patches, watermark, pub_key, config):
    N,g = pub_key
    
    for w_i, w_bit in enumerate(watermark):
        if w_bit == 0:
            continue

        patch = patches[w_i//3]
        axis = w_i % 3

        F_Nl = calculate_F_Nl(len(patch))
        T_Nl = calculate_T_Nl(len(patch), t=50)
        B_Nl = calculate_B_Nl(len(patch), F_Nl, T_Nl)
        B_Nl_c = paillier.encrypt_given_r(B_Nl, pub_key, 1)

        # # à garder pour vérification des valeurs
        # d = 0
        # for k in patch[1:]:
        #     d += paillier.decrypt(vertices[k][axis], config["encryption_keys"]["secret"], pub_key) - paillier.decrypt(vertices[patch[0]][axis], config["encryption_keys"]["secret"], pub_key)
        # if d > F_Nl :
        #     print("Probleme d > F(Nl)")


        pos_enc_direction = mpz(1)
        neg_enc_direction = mpz(1)

        for v in patch[1:]:
            pos_enc_direction *= (vertices[v][axis] * invert(vertices[patch[0]][axis], N**2)) % N**2
            neg_enc_direction *= (vertices[patch[0]][axis] * invert(vertices[v][axis], N**2)) % N**2

        mapped_direction = map_encrypted_direction((pos_enc_direction, neg_enc_direction), 2*F_Nl + T_Nl, pub_key)

        if mapped_direction > 0:
            for k in patch[1:] : vertices[k][axis] = (vertices[k][axis] * B_Nl_c) % N**2
        elif mapped_direction < 0:
            vertices[patch[0]][axis] = (vertices[patch[0]][axis] * B_Nl_c) % N**2

        # i = 0
        # mapped_direction = 0
        # while i < 2*F_Nl + T_Nl:
        #     i_c = powmod(1 + N, i, N**2)
        #     if i_c == pos_enc_direction % N**2:
        #         mapped_direction = i
        #         for k in patch[1:] : vertices[k][axis] = (vertices[k][axis] * B_Nl_c) % N**2
        #         break
        #     elif i_c == neg_enc_direction % N**2:
        #         mapped_direction = -i
        #         vertices[patch[0]][axis] = (vertices[patch[0]][axis] * B_Nl_c) % N**2
        #         break
        #     i += 1

        # # à garder pour vérification des valeurs
        # if d != mapped_direction:
        #     print(f"in embedding, patch[{w_i//3}][{axis%3}] : d={d}; mapped_d={mapped_direction}")

    return vertices

def map_encrypted_direction(encrypted_directions, direction_limit, pub_key) -> int:
    pos_direction, neg_direction = encrypted_directions
    N, g = pub_key
    i = 0
    while i < direction_limit:
        i_c = powmod(g, i, N**2)
        if i_c == pos_direction % N**2:
            return i
        elif i_c == neg_direction % N**2:
            return -i
        i += 1
    return


def extract(vertices, patches, encryption_keys, config):
    extracted_watermark = []
    pub_key = encryption_keys["public"]
    N, g = pub_key

    for w_i in range(config["message_length"]):
        patch = patches[w_i//3]
        axis = w_i % 3

        # F_Nl = 1.925*(len(patch) - 1)**3 - 60.6*(len(patch) - 1)**2 + 528*(len(patch) - 1) - 609
        F_Nl = calculate_F_Nl(len(patch))
        T_Nl = calculate_T_Nl(len(patch), t=50)
        B_Nl = calculate_B_Nl(len(patch), F_Nl, T_Nl)


        # # à garder pour vérification des valeurs
        # d = 0
        # for k in patch[1:]:
        #     d += paillier.decrypt(vertices[k][axis], config["encryption_keys"]["secret"], pub_key) - paillier.decrypt(vertices[patch[0]][axis], config["encryption_keys"]["secret"], pub_key)

        pos_enc_direction = mpz(1)
        neg_enc_direction = mpz(1)

        for v in patch[1:]:
            pos_enc_direction *= (vertices[v][axis] * invert(vertices[patch[0]][axis], N**2)) % N**2
            neg_enc_direction *= (vertices[patch[0]][axis] * invert(vertices[v][axis], N**2)) % N**2

        mapped_direction = map_encrypted_direction((pos_enc_direction, neg_enc_direction), 3*F_Nl + 2*T_Nl, pub_key)
        # i = 0
        # while i < 4*F_Nl + 2*T_Nl:
        #     i_c = powmod(1 + N, i, N**2)
        #     if i_c == pos_enc_direction % N**2:
        #         mapped_direction = i
        #         break
        #     elif i_c == neg_enc_direction % N**2:
        #         mapped_direction = -i
        #         break
        #     i += 1

        # à garder pour vérification des valeurs
        # if d != mapped_direction:
        #     print(f"in extraction, patch[{w_i//3}][{axis%3}] : d={d}; mapped_d={mapped_direction}")

        if -F_Nl < mapped_direction < F_Nl :
            extracted_watermark.append(0)
        else : 
            extracted_watermark.append(1)

    return extracted_watermark

def restore_encrypted_vertices(vertices, extracted_watermark, patch_indices, pub_key, config):
    N,g = pub_key
    for w_i in range(config["message_length"]):
        if extracted_watermark[w_i] == 0:
            continue
        patch = patch_indices[w_i//3]
        axis = w_i % 3

        F_Nl = calculate_F_Nl(len(patch))
        T_Nl = calculate_T_Nl(len(patch), t=5)
        B_Nl = calculate_B_Nl(len(patch), F_Nl, T_Nl)
        B_Nl_c = paillier.encrypt_given_r(B_Nl, pub_key, 1)

        pos_enc_direction = mpz(1)
        neg_enc_direction = mpz(1)

        for v in patch[1:]:
            pos_enc_direction *= (vertices[v][axis] * invert(vertices[patch[0]][axis], N**2)) % N**2
            neg_enc_direction *= (vertices[patch[0]][axis] * invert(vertices[v][axis], N**2)) % N**2

        mapped_direction = map_encrypted_direction((pos_enc_direction, neg_enc_direction), 3*F_Nl + 2*T_Nl, pub_key)
        # mapped_direction = 0
        # i = 0
        # while i < 4*F_Nl + 2*T_Nl:
        #     i_c = powmod(1 + N, i, N**2)
        #     if i_c == pos_enc_direction % N**2:
        #         mapped_direction = i
        #         break
        #     elif i_c == neg_enc_direction % N**2:
        #         mapped_direction = -i
        #         break
        #     i += 1

        if mapped_direction > F_Nl + T_Nl:
            for k in patch[1:] : vertices[k][axis] = (vertices[k][axis] * invert(B_Nl_c, N**2)) % N**2
        elif mapped_direction < -F_Nl - T_Nl:
            vertices[patch[0]][axis] = (vertices[patch[0]][axis] * invert(B_Nl_c, N**2)) % N**2

    return vertices

def inverse_preprocess_vertices(vertices_positive, preprocessing_info):
    """
    Inverse le preprocessing pour retrouver les coordonnées originales.
    
    Args:
        vertices_positive: vertices avec coordonnées entières positives
        preprocessing_info: dict contenant les infos du preprocessing
        
    Returns:
        vertices: coordonnées originales
    """
    k = preprocessing_info['k']
    
    vertices_normalized = _dequantization(vertices_positive, k)
    
    # 3. Dénormaliser si nécessaire
    if preprocessing_info['normalize']:
        vertices = _denormalize_vertices(vertices_normalized, 
                                       preprocessing_info['normalization_params'])
    else:
        vertices = vertices_normalized
    
    return vertices

def _dequantization(vertices_quantified, k=4):
    """
    Déquantifie les coordonnées des vertices en retrouvant les valeurs d'origine.

    Args:
        vertices_quantified: array numpy avec coordonnées quantifiées
        k: nombre de chiffres significatifs à conserver

    Returns:
        vertices: array numpy avec les coordonnées originales
    """
    # 1. Retirer 10^k
    vertices_int = vertices_quantified - 10**k

    # 2. Diviser par 10^k pour retrouver les décimales
    vertices = vertices_int.astype(float) / (10**k)

    return vertices

def _denormalize_vertices(vertices_normalized, normalization_params):
    """
    Inverse la normalisation pour retrouver les coordonnées originales.
    
    Args:
        vertices_normalized: vertices normalisés dans [0, 1]
        normalization_params: paramètres de normalisation
        
    Returns:
        vertices: coordonnées originales
    """
    v_min = normalization_params['v_min']
    v_range = normalization_params['v_range']
    
    vertices = vertices_normalized * v_range + v_min
    
    return vertices

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
    return watermarked_patches, nb_watermarked_bits

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
    F_Nl = calculate_F_Nl(Nl)
    T_Nl = calculate_T_Nl(Nl)

    params = {
        'F_Nl': F_Nl,
        'T_Nl': T_Nl,
        'B_Nl': calculate_B_Nl(Nl, F_Nl, T_Nl)
    }
    
    # Calculer les directions originales
    directions_encrypted = compute_all_directions_encrypted(encrypted_patch, N)
    directions = determine_directions_from_encrypted(directions_encrypted, N, Nl, k)
    
    # Tatouer chaque direction
    watermarked_encrypted_patch = encrypted_patch
    for j in range(3):
        if j < len(watermark_bits):
            bit = watermark_bits[j]
            direction = directions[j]
            watermarked_encrypted_patch = embed_bit_in_patch(watermarked_encrypted_patch, bit, j, direction, params, N)
    
    return watermarked_encrypted_patch

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

def preprocess_vertices(vertices, k=4):
    """
    Prétraite des vertices 3D pour le chiffrement Paillier.
    Normalise si nécessaire puis convertit en entiers positifs.
    
    Args:
        vertices: array numpy de shape (N, 3) avec les coordonnées
        k: nombre de chiffres significatifs à conserver (par défaut 4)
        
    Returns:
        vertices_positive: array numpy avec coordonnées entières positives
        preprocessing_info: dict contenant les infos pour l'inversion
    """
    preprocessing_info = {'k': k, 'normalize': False}

    vertices_positive, norm_params = _normalize_vertices(vertices)
    preprocessing_info['normalization_params'] = norm_params
    preprocessing_info['normalize'] = True
    vertices_quantified = _quantisation(vertices_positive, k)
    return vertices_quantified, preprocessing_info

def _normalize_vertices(vertices):
    """
    Normalise les vertices dans l'intervalle [0, 1].
    
    Args:
        vertices: array numpy de shape (N, 3) avec les coordonnées des vertices
        
    Returns:
        vertices_normalized: vertices normalisés dans [0, 1]
        normalization_params: dict contenant les paramètres pour l'inversion
    """
    # Calcul des min et max pour chaque dimension
    v_min = np.min(vertices, axis=0)
    v_max = np.max(vertices, axis=0)
    
    # Éviter la division par zéro
    v_range = v_max - v_min
    #assert v_range == 0, "le modèle a ses valeurs de coordonnées tous égaux "
    
    # Normalisation dans [0, 1]
    vertices_normalized = (vertices - v_min) / v_range
    
    # Sauvegarder les paramètres pour l'inversion
    normalization_params = {
        'v_min': v_min,
        'v_max': v_max,
        'v_range': v_range
    }
    
    return vertices_normalized, normalization_params

def _quantisation(vertices, k=4):
    """
    Quantifie les coordonnées des vertices en conservant k chiffres significatifs.

    Args:
        vertices: array numpy de shape (N, 3) avec les coordonnées
        k: nombre de chiffres significatifs à conserver

    Returns:
        vertices_quantified: array numpy avec coordonnées quantifiées
    """
    vertices_work = vertices.copy()
    
    # 1. Conversion en entiers (multiplication par 10^k)
    vertices_int = np.round(vertices_work * (10**k)).astype(int)
    # 2. Conversion en entiers positifs (ajout de 10^k)
    vertices_quantified = vertices_int + 10**k

    return vertices_quantified

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
        r = paillier.generate_r(N)
    
    # Chiffrer chaque vertex du patch
    encrypted_patch = []
    for vertex in patch:
        encrypted_vertex = _encrypt_vertex(vertex, pub_key, r)
        encrypted_patch.append(encrypted_vertex)
    
    return encrypted_patch, r

def _encrypt_vertex(vertex_coords, pub_key, r):
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

def divide_into_patches(vertices, faces):
    """
    Divise un modèle 3D en patches non-chevauchants.
    
    Args:
        vertices: array numpy (N, 3) des coordonnées
        faces: array numpy (F, 3) des indices des faces
        
    Returns:
        patches: liste d'arrays numpy des patches
        patches_indices: liste des indices pour chaque patch
        isolated_indices: liste des indices des vertices isolés
        isolated_coords: liste des coordonnées des vertices isolés
    """
    n_vertices = len(vertices)
    adjacency = _build_adjacency_graph(faces, n_vertices)
    
    unclassified = set(range(n_vertices))
    classified = set()
    patches_indices = []
    isolated_indices = []
    
    while unclassified:
        # Sélectionner le premier vertex non classé
        seed = min(unclassified)
        
        # Former le patch avec son 2-ring neighborhood complet
        patch_vertices = _get_k_ring_neighbors(seed, adjacency,k=2)
        patch_vertices = patch_vertices - classified
        
        # S'assurer que le patch a au moins 2 vertices
        if len(patch_vertices) >= 2:
            # Créer le patch avec le vertex central en premier
            patch_idx = [seed] + sorted([v for v in patch_vertices if v != seed])
            patches_indices.append(patch_idx)
            
            # Marquer ces vertices comme classés
            classified.update(patch_vertices)
            unclassified -= patch_vertices
        else:
            # Vertex isolé
            isolated_indices.append(seed)
            unclassified.remove(seed)
            classified.add(seed)
    
    # Convertir les indices en arrays de coordonnées
    patches = []
    if patches_indices:
        for indices in patches_indices:
                patch = vertices[indices]
                patches.append(patch)
    
    isolated_coords = []
    if isolated_indices:
        for indices in isolated_indices:
            isolated_coords.append(vertices[indices])
            
    return (patches, patches_indices), (isolated_coords, isolated_indices)

def get_patch_info(patches, isolated_coords=None):
    """
    Obtient des informations sur les patches.
    
    Args:
        patches: liste de patches
        isolated_coords: liste des coordonnées des vertices isolés
        
    Returns:
        dict: statistiques des patches
    """
    sizes = [len(patch) for patch in patches]
    
    info = {
        'n_patches': len(patches),
        'min_size': min(sizes) if sizes else 0,
        'max_size': max(sizes) if sizes else 0,
        'avg_size': np.mean(sizes) if sizes else 0,
        'sizes': sizes,
        'n_isolated': len(isolated_coords) if isolated_coords else 0
    }
    
    return info

def _build_adjacency_graph(faces, n_vertices):
    """
    Construit le graphe d'adjacence à partir des faces.
    
    Args:
        faces: array numpy (F, 3) des indices des faces (indices commençant à 1)
        n_vertices: nombre total de vertices
        
    Returns:
        list: liste d'adjacence pour chaque vertex
    """
    adjacency = [set() for _ in range(n_vertices)]
    
    # Pour chaque face, connecter tous les vertices
    for face in faces:
        # Convertir les indices 1-based en 0-based
        # v1, v2, v3 = face[0] - 1, face[1] - 1, face[2] - 1
        v1, v2, v3 = face[0], face[1], face[2] # MODIFIÉ
        adjacency[v1].update([v2, v3])
        adjacency[v2].update([v1, v3])
        adjacency[v3].update([v1, v2])
    
    return adjacency

def _get_k_ring_neighbors(vertex_idx, adjacency, k=2):
    """
    Obtient le voisinage de rang k d'un vertex.
    
    Args:
        vertex_idx: index du vertex central
        adjacency: graphe d'adjacence
        n: rang du voisinage (1-ring, 2-ring, etc.)
        
    Returns:
        set: indices des vertices dans le k-ring neighborhood
    """
    if k == 0:
        return {vertex_idx}
    
    neighbors = {vertex_idx}
    current_ring = {vertex_idx}
    
    for _ in range(k):
        next_ring = set()
        for v in current_ring:
            next_ring.update(adjacency[v])
        current_ring = next_ring - neighbors
        neighbors.update(next_ring)
    
    return neighbors

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
        encrypted_vertex= _encrypt_vertex(vertex, pub_key,r)
        encrypted_isolated_coords.append(encrypted_vertex)
    return encrypted_isolated_coords

def calculate_F_Nl(Nl):
    # F_Nl = 1.925*(Nl - 1)**3 - 60.6*(Nl - 1)**2 + 528*(Nl - 1) - 609
    F_Nl = (Nl-1) * 10000/2

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


def calculate_B_Nl(Nl, F_Nl, T_Nl):
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
    
    B_Nl = ceil((F_Nl + T_Nl) / (Nl - 1))
    return B_Nl

def recover_encrypted_model(restored_encrypted_patches, patches_indices, encrypted_isolated_coords, isolated_indices, n_vertices):
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