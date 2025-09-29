"""
preprocessing.py - Module de preprocessing pour RRDH-ED améliorée
Convertit les coordonnées décimales en entiers positifs pour le chiffrement Paillier
"""

import numpy as np


def normalize_vertices(vertices):
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


def denormalize_vertices(vertices_normalized, normalization_params):
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


def quantification(vertices, k=4):
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

def dequantification(vertices_quantified, k=4):
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
    
    # Normaliser si les valeurs sont hors de [-1, 1]
    vertices_work, norm_params = normalize_vertices(vertices)
    preprocessing_info['normalization_params'] = norm_params
    preprocessing_info['normalize'] = True
    vertices_positive = quantification(vertices_work, k)

    return vertices_positive, preprocessing_info


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
    
    vertices_normalized = dequantification(vertices_positive, k)
    
    # 3. Dénormaliser si nécessaire
    if preprocessing_info['normalize']:
        vertices = denormalize_vertices(vertices_normalized, 
                                       preprocessing_info['normalization_params'])
    else:
        vertices = vertices_normalized
    
    return vertices