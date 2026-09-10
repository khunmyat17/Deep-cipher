# Deep Cipher

Permanent web deployment for the Deep Cipher project.

Routes:

- `/sender` — sender-side GUI
- `/receiver` — receiver-side GUI
- `/setup` — one-time model checkpoint upload
- `/health` — Railway healthcheck

## Required Railway variable

Set:

`SETUP_TOKEN=<your private setup password>`

Recommended:

`DATA_DIR=/data`

`MODEL_PATH=/data/deep_cipher_final.pth`

## Model storage

Attach a Railway persistent volume mounted at `/data`.

After deployment, open `/setup` and upload:

`deep_cipher_final.pth`

## Important

The generated stego output is 16 kHz mono Float32 WAV.
Do not convert, normalize, or edit it before receiver decoding.
