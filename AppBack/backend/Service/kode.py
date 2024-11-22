import os
secret_key = os.urandom(24).hex()  # menghasilkan secret key dengan panjang 48 karakter (hex)
print(secret_key)
