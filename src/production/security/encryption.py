"""
[ELITE ARCHITECTURE] encryption.py
AES-256 Storage Encryption for Knowledge Base.
"""

from cryptography.fernet import Fernet
import os
from loguru import logger

class VaultManager:
    """
    Innovation: Data Sovereignty.
    Encrypts sensitive research data at rest to prevent unauthorized 
    access to the document vault.
    """
    
    def __init__(self, key_path: str = "config/vault.key"):
        if os.path.exists(key_path):
            with open(key_path, "rb") as f:
                self.key = f.read()
        else:
            self.key = Fernet.generate_key()
            os.makedirs(os.path.dirname(key_path), exist_ok=True)
            with open(key_path, "wb") as f:
                f.write(self.key)
            logger.info("New encryption key generated.")
            
        self.cipher = Fernet(self.key)

    def encrypt_data(self, data: str) -> bytes:
        return self.cipher.encrypt(data.encode())

    def decrypt_data(self, encrypted_data: bytes) -> str:
        return self.cipher.decrypt(encrypted_data).decode()

if __name__ == "__main__":
    vault = VaultManager()
    token = vault.encrypt_data("Sensitive Document Content")
    print(f"Encrypted: {token}")
    print(f"Decrypted: {vault.decrypt_data(token)}")
