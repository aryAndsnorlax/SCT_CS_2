# 🔒 Simple Image Encryptor

A simple Python tool to *encrypt and decrypt images* using pixel manipulation techniques.

## ✨ Features
- Supported operations:
  - *xor* → XOR each pixel byte with a keystream  
  - *add* → Add a keystream (mod 256) to each pixel byte (inverse = subtract)  
  - *swap* → Shuffle pixel bytes with a PRNG permutation  
  - *mix* → A combo: swap → xor → add (default, strongest)

- Works with any Pillow-readable image (JPG, PNG, etc.)
- Saves output as *PNG* to avoid quality loss
- Deterministic: encryption/decryption requires the same *key* and *nonce*

---

## 🛠 Installation



Install dependencies:

bash
pip install -r requirements.txt


---

## 🚀 Usage

### Encrypt an image
bash
python simple_image_encryptor.py encrypt input.jpg enc.png --mode mix --key "mypassword" --nonce 42


### Decrypt an image
bash
python simple_image_encryptor.py decrypt enc.png dec.png --mode mix --key "mypassword" --nonce 42


---

## 📌 Notes
- You *must* use the same --key and --nonce for decryption.
- Default mode is mix (best visual scrambling).
- Not intended as a replacement for real cryptography (AES, etc.). This is for *educational/demo purposes*.

---

## 📂 Example
bash
# XOR encryption
python simple_image_encryptor.py encrypt photo.png enc.png --mode xor --key "secret" --nonce 99

# Decrypt back
python simple_image_encryptor.py decrypt enc.png dec.png --mode xor --key "secret" --nonce 99


---

