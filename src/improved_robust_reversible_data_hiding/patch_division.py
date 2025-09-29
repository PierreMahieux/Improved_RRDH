"""
patch_division.py - Module de division en patches pour RRDH-ED améliorée
Divise un modèle 3D en patches non-chevauchants en utilisant l'algorithme 2-ring neighborhood
"""

import numpy as np


def build_adjacency_graph(faces, n_vertices):
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
        v1, v2, v3 = face[0] - 1, face[1] - 1, face[2] - 1
        adjacency[v1].update([v2, v3])
        adjacency[v2].update([v1, v3])
        adjacency[v3].update([v1, v2])
    
    return adjacency


def get_k_ring_neighbors(vertex_idx, adjacency, k=2):
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
    adjacency = build_adjacency_graph(faces, n_vertices)
    
    unclassified = set(range(n_vertices))
    classified = set()
    patches_indices = []
    isolated_indices = []
    
    while unclassified:
        # Sélectionner le premier vertex non classé
        seed = min(unclassified)
        
        # Former le patch avec son 2-ring neighborhood complet
        patch_vertices = get_k_ring_neighbors(seed, adjacency,k=2)
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


def reconstruct_model(patches, patches_indices, isolated_coords, isolated_indices, n_vertices):
    """
    Reconstruit le modèle complet à partir des patches et vertices isolés.
    
    Args:
        patches: liste d'arrays numpy des patches déchiffrés ou en clairs
        patches_indices: indices originaux des patches
        isolated_coords: liste des coordonnées des vertices isolés en clair
        isolated_indices: listes des indices des vertices isolés 
        n_vertices: nombre total de vertices
        
    Returns:
        vertices: modèle reconstruit
    """
    vertices = np.zeros((n_vertices, 3))
    
    # Replacer les patches
    for patch, indices in zip(patches, patches_indices):
        vertices[indices] = patch
    
    # Replacer les vertices isolés
    if isolated_indices and isolated_coords:
        vertices[isolated_indices] = isolated_coords
    
    return vertices


