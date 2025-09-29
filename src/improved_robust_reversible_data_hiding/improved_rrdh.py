"""
main.py - Workflow principal RRDH-ED (Robust Reversible Data Hiding in Encrypted Domain)
Focus sur le domaine chiffré uniquement
"""

import numpy as np
import sys

from datetime import datetime
import os
import time
from gmpy2 import mpz, powmod, invert
# Imports des modules RRDH-ED
from src.improved_robust_reversible_data_hiding.preprocessing import preprocess_vertices, inverse_preprocess_vertices
from src.improved_robust_reversible_data_hiding.patch_division import divide_into_patches, get_patch_info
from src.improved_robust_reversible_data_hiding.encryption import (
    generate_keys_for_rrdh, encrypt_patches,encrypt_isolated_vertices, encrypt_vertex
)
from src.improved_robust_reversible_data_hiding.watermarking import embed_watermark_in_model
from src.improved_robust_reversible_data_hiding.extraction_restoration_ED import (
    extract_watermark_from_model, restore_encrypted_patches_from_watermarking,
    reconstruct_encrypted_model, decrypt_complete_model, extract_bit_from_direction
)
from src.improved_robust_reversible_data_hiding import directions 
from src.utils import paillier, util, mesh_utils

def run(config: dict, encryption_keys:dict, watermarks: tuple, model):
    print("Run Bamba RRDH")
    pub_key = encryption_keys["public"]
    priv_key = encryption_keys["secret"]

    N, g = pub_key

    vertices = model["vertices"]
    faces = model["faces"]
    n_vertices = len(vertices)

    watermark = watermarks[0]

    result = {"config": config}

    # Preprocessing
    vertices_prep, prep_info = preprocess_vertices(vertices, k=4)
    mesh_utils.save_3d_model(vertices_prep, faces, os.path.join(config["result_folder"],"preprocessed.obj"))

    # 2. DIVISION EN PATCHES
    print("\n2. Division en patches...")
    (patches, patch_indices), (isolated_coords, isolated_indices) = divide_into_patches(vertices_prep, faces)
    patch_info = get_patch_info(patches, isolated_coords)

    result["patch_number"] = len(patch_indices)
    result["max_vertices_per_patch"] = max([len(patch_indices[i]) for i in range(len(patch_indices))])
    result["min_vertices_per_patch"] = min([len(patch_indices[i]) for i in range(len(patch_indices))])
    
    # 3. CHIFFREMENT DES PATCHES
    print("\n3. Chiffrement des patches...")
    start_encryption = time.time()
    encrypted_patches, r_values = encrypt_patches(patches, pub_key)
    encrypted_isolated = encrypt_isolated_vertices(isolated_coords, pub_key) if isolated_coords else []
    #recosntruction du modèle chiffré complet
    # encrypted_vertices = reconstruct_encrypted_model(
    #     encrypted_patches, patch_indices, encrypted_isolated, isolated_indices, n_vertices
    # )

    encrypted_vertices = []
    for v in vertices_prep:
        encrypted_vertices.append(encrypt_vertex(v, pub_key, 1)) # r fixe
    
    # 5. TATOUAGE DANS LE DOMAINE CHIFFRÉ
    print("\n5. Tatouage dans le domaine chiffré...")
    start_embedding = time.time()
    # watermarked_patches_old, nb_watermaked_bits = embed_watermark_in_model(
    #     encrypted_patches, watermark, N, k=prep_info['k']
    # )

    watermarked_patches = encrypted_patches.copy()

    watermarked_encrypted_vertices = encrypted_vertices.copy()
    for w_i in range(len(watermark)):
        if watermark[w_i] == 0:
            continue

        patch = patch_indices[w_i//3]
        patch_vertices = [encrypted_vertices[i] for i in patch]
        axis = w_i % 3
        
        Nl = len(patch)
        params = {
            'F_Nl': directions.calculate_F_Nl(Nl, prep_info['k']),
            'T_Nl': directions.calculate_T_Nl(Nl),
            'B_Nl': directions.calculate_B_Nl(Nl, t=50, k=prep_info['k'])
        }

        encrypted_direction = directions.compute_encrypted_direction(patch_vertices, axis, N)
        direction = directions.calculate_direction_from_encrypted(encrypted_direction, N, params["F_Nl"])

        g_B = powmod(g, params["B_Nl"], N**2)

        if direction >= 0:
            for k in patch[1:]:
                watermarked_encrypted_vertices[k][axis] = (watermarked_encrypted_vertices[k][axis] * g_B) % N**2
        elif direction < 0:
            watermarked_encrypted_vertices[patch[0]][axis] = (watermarked_encrypted_vertices[patch[0]][axis] * g_B) % N**2

    result["time_embedding"] = time.time() - start_embedding
    
    # Reconstruction des vertices chiffrés tatoués
    # watermarked_encrypted_vertices = reconstruct_encrypted_model(
    #     watermarked_patches, patch_indices, encrypted_isolated, isolated_indices, n_vertices
    # )
    
    # 6. EXTRACTION DANS LE DOMAINE CHIFFRÉ
    print("\n6. Extraction dans le domaine chiffré...")
    start_extraction = time.time()
    # extracted_watermark = extract_watermark_from_model(
    #     watermarked_patches, N, 
    #     expected_length=config["message_length"],
    #     k=prep_info['k']
    # )

    extracted_watermark = []
    for w_i in range(config["message_length"]):
        patch = patch_indices[w_i//3]
        patch_vertices = [watermarked_encrypted_vertices[i] for i in patch]
        axis = w_i % 3

        Nl = len(patch)
        F_Nl = directions.calculate_F_Nl(Nl,config["quantisation_factor"])
        T_Nl = directions.calculate_T_Nl(Nl)

        F_limit = 2 * F_Nl + T_Nl

        encrypted_direction = directions.compute_encrypted_direction(patch_vertices, axis, N)
        direction = directions.calculate_direction_from_encrypted(encrypted_direction, N, F_limit)
        extracted_watermark.append(extract_bit_from_direction(direction, F_Nl, T_Nl))

    result["time_extraction"] = time.time() - start_extraction
    # Calcul du BER
    result["BER"] = util.compare_bits(watermark, extracted_watermark)

    
    # 7. RESTAURATION DANS LE DOMAINE CHIFFRÉ
    print("\n7. Restauration dans le domaine chiffré...")
    
    # restored_encrypted_patches = restore_encrypted_patches_from_watermarking(watermarked_patches, N, k=prep_info['k'])
    
    # Reconstruction des vertices chiffrés restaurés
    # restored_encrypted_vertices = reconstruct_encrypted_model(
    #     restored_encrypted_patches, patch_indices, encrypted_isolated, isolated_indices, n_vertices
    # )

    restored_encrypted_vertices = watermarked_encrypted_vertices.copy()
    for w_i in range(len(extracted_watermark)):
        if extracted_watermark[w_i] == 0:
            continue

        patch = patch_indices[w_i//3]
        patch_vertices = [restored_encrypted_vertices[i] for i in patch]
        axis = w_i % 3

        Nl = len(patch)
        F_Nl = directions.calculate_F_Nl(Nl,config["quantisation_factor"])
        T_Nl = directions.calculate_T_Nl(Nl)
        B_Nl = directions.calculate_B_Nl(Nl, t=50, k=config["quantisation_factor"])
        
        g = N + 1
        
        # Calculer l'inverse de g^B(Nl)
        g_B = powmod(g, B_Nl, N**2)
        theta_g_B = invert(g_B, N**2)

        F_limit = 2 * F_Nl + T_Nl
        encrypted_direction = directions.compute_encrypted_direction(patch_vertices, axis, N)
        direction = directions.calculate_direction_from_encrypted(encrypted_direction, N, F_limit)
        
        threshold = F_Nl + T_Nl / 2

        if direction >= 0:
            for k in patch[1:]:
                restored_encrypted_vertices[k][axis] = (restored_encrypted_vertices[k][axis] * theta_g_B) % N**2
        else :
            restored_encrypted_vertices[patch[0]][axis] = (restored_encrypted_vertices[patch[0]][axis] * theta_g_B) % N**2



    # Vertices restaurés déchiffrés
    restored_decrypted_vertices = decrypt_complete_model(restored_encrypted_vertices, priv_key, pub_key)
    # Restauration compléte en appliquant l'inverse du preprocessing
    restored_clear = inverse_preprocess_vertices(restored_decrypted_vertices, prep_info)

    mesh_utils.save_3d_model(restored_clear, faces, os.path.join(config["result_folder"],"fully_restored.obj"))

    #
    
    # 8. Modèle déchiffré tatoué
    print("\n8. Modèle déchiffré tatoué...")
    
    watermarked_decrypted_vertices = np.array(watermarked_encrypted_vertices.copy())
    for v_i in range(len(watermarked_decrypted_vertices)):
        for c_i in range(3):
            watermarked_decrypted_vertices[v_i][c_i] = int(paillier.decrypt_CRT(watermarked_encrypted_vertices[v_i][c_i], priv_key, pub_key))
    # Inverse preprocessing
    watermarked_clear = inverse_preprocess_vertices(watermarked_decrypted_vertices, prep_info)
    mesh_utils.save_3d_model(watermarked_clear, faces, os.path.join(config["result_folder"],"watermarked_restored.obj"))
    
    return result

def main(obj_filename):
    print("RRDH-ED: Robust Reversible Data Hiding in Encrypted Domain")
    script_dir = os.path.dirname(__file__)
    # 1. CHARGEMENT ET PREPROCESSING
    print("\n1. Chargement du modèle 3D...")
    vertices, faces = mesh_utils.load_3d_model(obj_filename)
    n_vertices = len(vertices)
    # Création du répertoire de sortie
    result_folder = os.path.join(script_dir, f"./results/{obj_filename.split('/')[-1].split(".")[0]}/")

    config = {"key_size": 256, "quantisation_factor": 4, "result_folder": result_folder}
    result ={"config": config}

    # Preprocessing
    vertices_prep, prep_info = preprocess_vertices(vertices, k=4)

    # 2. DIVISION EN PATCHES
    print("\n2. Division en patches...")
    (patches, patch_indices), (isolated_coords, isolated_indices) = divide_into_patches(vertices_prep, faces)
    patch_info = get_patch_info(patches, isolated_coords)

    #génération des clés
    pub_key, priv_key = generate_keys_for_rrdh(256)
    N, g = pub_key

    # 3. CHIFFREMENT DES PATCHES
    print("\n3. Chiffrement des patches...")
    start_encryption = time.time()
    encrypted_patches, r_values = encrypt_patches(patches, pub_key)
    encrypted_isolated = encrypt_isolated_vertices(isolated_coords, pub_key) if isolated_coords else []
    #recosntruction du modèle chiffré complet
    encrypted_vertices = reconstruct_encrypted_model(
        encrypted_patches, patch_indices, encrypted_isolated, isolated_indices, n_vertices
    )
    
    # 4. GÉNÉRATION DU WATERMARK
    print("\n4. Génération du watermark...")
    watermark_length = patch_info['n_patches'] * 3
    watermark_original = [np.random.randint(0, 2) for _ in range(watermark_length)]

    
    # 5. TATOUAGE DANS LE DOMAINE CHIFFRÉ
    print("\n5. Tatouage dans le domaine chiffré...")
    start_embedding = time.time()
    watermarked_patches, nb_watermaked_bits = embed_watermark_in_model(
        encrypted_patches, watermark_original, N, k=prep_info['k']
    )
    result["time_embedding"] = time.time() - start_embedding
    print(f" Nombres de Patches tatoués: {nb_watermaked_bits//3}, Nombre de Bits tatoués: {nb_watermaked_bits}")
    # Reconstruction des vertices chiffrés tatoués
    watermarked_encrypted_vertices = reconstruct_encrypted_model(
        watermarked_patches, patch_indices, encrypted_isolated, isolated_indices, n_vertices
    )
    
    # 6. EXTRACTION DANS LE DOMAINE CHIFFRÉ
    print("\n6. Extraction dans le domaine chiffré...")
    start_extraction = time.time()
    extracted_watermark = extract_watermark_from_model(
        watermarked_patches, N, 
        expected_length=watermark_length,
        k=prep_info['k']
    )
    result["time_extraction"] = time.time() - start_extraction
    # Calcul du BER
    result["BER"] = util.compare_bits(watermark_original, extracted_watermark)

    
    # 7. RESTAURATION DANS LE DOMAINE CHIFFRÉ
    print("\n7. Restauration dans le domaine chiffré...")
    
    restored_encrypted_patches = restore_encrypted_patches_from_watermarking(watermarked_patches, N, k=prep_info['k'])
    
    # Reconstruction des vertices chiffrés restaurés
    restored_encrypted_vertices = reconstruct_encrypted_model(
        restored_encrypted_patches, patch_indices, encrypted_isolated, isolated_indices, n_vertices
    )
    # Vertices restaurés déchiffrés
    restored_decrypted_vertices = decrypt_complete_model(restored_encrypted_vertices, priv_key, pub_key)
    # Restauration compléte en appliquant l'inverse du preprocessing
    restored_clear = inverse_preprocess_vertices(restored_decrypted_vertices, prep_info)

    #
    
    # 8. Modèle déchiffré tatoué
    print("\n8. Modèle déchiffré tatoué...")
    
    watermarked_decrypted_vertices = decrypt_complete_model(watermarked_encrypted_vertices, priv_key, pub_key)
    # Inverse preprocessing
    watermarked_clear = inverse_preprocess_vertices(watermarked_decrypted_vertices, prep_info)


    # Sauvegarder les modèles
    mesh_utils.save_3d_model(vertices, faces, os.path.join(config["result_folder"],"original.obj"))
    mesh_utils.save_3d_model(vertices_prep, faces, os.path.join(config["result_folder"],"preprocessed.obj"))
    mesh_utils.save_3d_model(watermarked_clear, faces, os.path.join(config["result_folder"],"watermarked_decrypted.obj"))
    mesh_utils.save_3d_model(restored_clear, faces, os.path.join(config["result_folder"],"restored_decrypted.obj"))
    mesh_utils.save_3d_model(restored_decrypted_vertices, faces, os.path.join(config["result_folder"],"restored_decrypted_prep.obj"))


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python main.py chemin/vers/modele.obj")
    # else:
    try:
        list_obj_paths = "/home/pierremahieux/Documents/Models/Casting/casting.obj"
        main("sys.argv[1]")
    except Exception as e:
        print(f"\nErreur: {e}")
        import traceback
        traceback.print_exc()