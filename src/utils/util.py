import hashlib
import os
import uuid
import datetime
import json
import ecdsa


def genereate_signing_keys() -> dict:
    sk = ecdsa.SigningKey.generate(curve=ecdsa.NIST256p)
    vk = sk.verifying_key
    return {"signing": sk, "verifying": vk}

def verify_signature(signature, message: bytes, verifying_key) -> bool:
    try:
        verifying_key.verify(signature, message)
        return True
    except ecdsa.BadSignatureError:
        return False

def generate_signature(message: bytes, signing_key) -> bytes:
    return signing_key.sign(message)

def hash_string_to_bits(message: str):
    """
    Génère 256 bits de tatouage à partir d'un hash SHA-256
    
    message: le message à tatouer (str). Si None, génère un message aléatoire
    
    Retourne le hash du message en 256 bits 
    """
    assert(message!=None)
    
    # Calculer le SHA-256
    hash_bytes = hashlib.sha256(message.encode('utf-8')).digest()
    
    return bytes_to_bits(hash_bytes)

def int_to_bits(value, num_bits):
    """
    Convertit un entier en liste de bits
    
    value: entier à convertir
    num_bits: nombre de bits souhaité
             
    Retourne la liste de bits [MSB, ..., LSB]
    """
    # S'assurer que value est un entier Python
    value = int(value)
    
    bits = []
    for i in range(num_bits):
        bits.append((value >> (num_bits - 1 - i)) & 1)
    return bits

def bits_to_int(bits):
    """
    Convertit une liste de bits en entier
    
    bits: liste de bits [MSB, ..., LSB]
    
    Retourne l'entier correspondant
    """
    value = 0
    for i, bit in enumerate(reversed(bits)):
        # Convertir en int Python pour éviter les problèmes numpy
        value += int(bit) * (2 ** i)
    return int(value)

def bytes_to_bits(byte_message: bytes) -> list:
    bits = []
    for byte in byte_message:
        binary_string = format(byte, '08b') 
        for bit_char in binary_string:
            bits.append(int(bit_char))
    
    return bits

def bits_to_bytes(bits_message: list) -> bytes:
    s = ''.join(str(x) for x in bits_message)[0::]
    return int(s, 2).to_bytes(len(s) // 8, byteorder='big')

def write_report(results: dict) -> None:
    try:
        os.makedirs(os.path.dirname(results["config"]["result_folder"]), exist_ok=True)
        filename = os.path.join(results["config"]["result_folder"], "report.txt")
        with open(filename, 'w') as file:
            for key, value in results.items():
                if "model" not in key and "signature" not in key and "watermark" not in key and "blocks" not in key and "keys" not in key:
                    file.write(f"\"{key}\": {value},\n")
                # if {"model", "vertices", "signature", "watermark", "blocks", "keys"} not in key:
                #     file.write(f"\"{key}\": {value},\n")
        
        print(f"Rapport sauvegardé dans {filename}")
    except Exception as e:
        print(f"Erreur lors de l'écriture du rapport': {e}")  

    return None

def compare_bits(original_bits, extracted_bits):
    """Compare deux séquences de bits et affiche les statistiques et retourne le BER [0,1]"""
    min_len = min(len(original_bits), len(extracted_bits))
    
    if min_len == 0:
        print("Erreur: Au moins une séquence est vide")
        return 1
    
    # Comparer bit par bit
    errors = 0
    for i in range(min_len):
        if original_bits[i] != extracted_bits[i]:
            errors += 1
    
    # Calculer le BER (Bit Error Rate)
    ber = errors / min_len
    
    return ber