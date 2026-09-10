import os
import math
import time
import struct
import secrets
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import librosa
from PIL import Image
import gradio as gr

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidTag


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(DATA_DIR / "deep_cipher_final.pth")
    )
)

OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SETUP_TOKEN = os.getenv("SETUP_TOKEN", "")

SAMPLE_RATE = 16000
SEGMENT_SECONDS = 3
AUDIO_LENGTH = SAMPLE_RATE * SEGMENT_SECONDS
SECRET_BITS = 128
CHUNK_BYTES = 16
MAX_MESSAGE_BYTES = 2000

ADAPTIVE_STRENGTHS = [
    0.04,
    0.05,
    0.06,
    0.075,
    0.10,
    0.125,
    0.15,
    0.175,
    0.20,
]

AES_KEY_BYTES = 32
SALT_BYTES = 16
NONCE_BYTES = 12
AES_TAG_BYTES = 16
PBKDF2_ITERATIONS = 200000

PACKET_MAGIC = b"DCMX"
PACKET_VERSION = 1
PACKET_HEADER_SIZE = 4 + 1 + SALT_BYTES + NONCE_BYTES + 4

PAYLOAD_MAGIC = b"PAY1"
PAYLOAD_HEADER_SIZE = 4 + 1 + 4 + 1 + 1 + 4

FLAG_MESSAGE = 1
FLAG_IMAGE = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL
# ============================================================

class DeepCipherV2(nn.Module):

    def __init__(self, secret_bits=128):

        super().__init__()

        self.secret_bits = secret_bits

        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),

            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
        )

        self.audio_decoder = nn.Sequential(
            nn.ConvTranspose1d(
                64, 32,
                kernel_size=16,
                stride=2,
                padding=7
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.ConvTranspose1d(
                32, 1,
                kernel_size=16,
                stride=2,
                padding=7
            ),
            nn.Tanh(),
        )

        self.secret_encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2),

            nn.Conv1d(8, 8, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2),
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(72, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),

            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.Tanh(),
        )

        self.embedding_strength = 0.04

        self.secret_decoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2),

            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2),

            nn.Conv1d(64, 64, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool1d(secret_bits),
            nn.Conv1d(64, 1, kernel_size=1),
        )

    def forward(self, audio, secret_bits):

        audio_latent = self.audio_encoder(audio)

        secret = secret_bits.unsqueeze(1)

        secret = F.interpolate(
            secret,
            size=audio_latent.shape[-1],
            mode="linear",
            align_corners=False
        )

        secret_features = self.secret_encoder(secret)

        combined = torch.cat(
            [audio_latent, secret_features],
            dim=1
        )

        secret_delta = self.fusion(combined)

        stego_latent = (
            audio_latent
            +
            self.embedding_strength
            *
            secret_delta
        )

        stego_audio = self.audio_decoder(
            stego_latent
        )

        secret_logits = (
            self.secret_decoder(stego_audio)
            .squeeze(1)
        )

        return stego_audio, secret_logits


_deep_cipher_model = None
_model_error = None


def load_model(force=False):

    global _deep_cipher_model
    global _model_error

    if (
        _deep_cipher_model is not None
        and
        not force
    ):
        return _deep_cipher_model

    if not MODEL_PATH.exists():

        _model_error = (
            f"Model not found at {MODEL_PATH}. "
            "Upload deep_cipher_final.pth at /setup."
        )

        return None

    try:

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=False
        )

        if isinstance(checkpoint, dict):

            secret_bits = int(
                checkpoint.get(
                    "secret_bits",
                    128
                )
            )

        else:

            secret_bits = 128

        if secret_bits != 128:

            raise ValueError(
                "This app expects a 128-bit model checkpoint."
            )

        model = DeepCipherV2(
            secret_bits=secret_bits
        ).to(DEVICE)

        if (
            isinstance(checkpoint, dict)
            and
            "model_state_dict" in checkpoint
        ):

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            state_dict = checkpoint

        model.load_state_dict(
            state_dict,
            strict=True
        )

        if isinstance(checkpoint, dict):

            model.embedding_strength = float(
                checkpoint.get(
                    "embedding_strength",
                    0.04
                )
            )

        model.eval()

        _deep_cipher_model = model
        _model_error = None

        return model

    except Exception as exc:

        _deep_cipher_model = None
        _model_error = f"Model load failed: {exc}"

        return None


def require_model():

    model = load_model()

    if model is None:

        raise RuntimeError(
            _model_error
            or
            "Deep Cipher model is not ready."
        )

    return model


# ============================================================
# MESSAGE + IMAGE PAYLOAD
# ============================================================

def parse_image_size(option):

    text = str(option or "16")

    if text.startswith("24"):
        return 24

    if text.startswith("8"):
        return 8

    return 16


def prepare_image(
    image_path,
    side
):

    if not image_path:

        return (
            b"",
            None,
            0,
            0
        )

    image = Image.open(
        image_path
    ).convert("L")

    image = image.resize(
        (side, side),
        Image.Resampling.LANCZOS
    )

    array = np.asarray(
        image,
        dtype=np.uint8
    )

    return (
        array.tobytes(),
        image,
        side,
        side
    )


def build_payload(
    message,
    image_path,
    image_size_option
):

    message = message or ""

    message_bytes = message.encode(
        "utf-8"
    )

    if len(message_bytes) > MAX_MESSAGE_BYTES:

        raise ValueError(
            f"Message too long. "
            f"Maximum is {MAX_MESSAGE_BYTES} UTF-8 bytes."
        )

    side = parse_image_size(
        image_size_option
    )

    (
        image_bytes,
        tiny_image,
        width,
        height
    ) = prepare_image(
        image_path,
        side
    )

    if (
        not message_bytes
        and
        not image_bytes
    ):

        raise ValueError(
            "Enter a message, upload an image, or both."
        )

    flags = 0

    if message_bytes:
        flags |= FLAG_MESSAGE

    if image_bytes:
        flags |= FLAG_IMAGE

    header = (
        PAYLOAD_MAGIC
        +
        bytes([flags])
        +
        struct.pack(
            ">I",
            len(message_bytes)
        )
        +
        bytes([width])
        +
        bytes([height])
        +
        struct.pack(
            ">I",
            len(image_bytes)
        )
    )

    payload = (
        header
        +
        message_bytes
        +
        image_bytes
    )

    return (
        payload,
        tiny_image,
        len(message_bytes),
        len(image_bytes),
        width,
        height
    )


def parse_payload(payload):

    if (
        len(payload) < PAYLOAD_HEADER_SIZE
        or
        payload[:4] != PAYLOAD_MAGIC
    ):

        raise ValueError(
            "Invalid combined payload."
        )

    offset = 4

    flags = payload[offset]
    offset += 1

    message_len = struct.unpack(
        ">I",
        payload[
            offset:
            offset + 4
        ]
    )[0]

    offset += 4

    width = payload[offset]
    offset += 1

    height = payload[offset]
    offset += 1

    image_len = struct.unpack(
        ">I",
        payload[
            offset:
            offset + 4
        ]
    )[0]

    offset += 4

    required_size = (
        PAYLOAD_HEADER_SIZE
        +
        message_len
        +
        image_len
    )

    if len(payload) < required_size:

        raise ValueError(
            "Recovered payload is incomplete."
        )

    message_bytes = payload[
        offset:
        offset + message_len
    ]

    offset += message_len

    image_bytes = payload[
        offset:
        offset + image_len
    ]

    if flags & FLAG_MESSAGE:

        message = message_bytes.decode(
            "utf-8"
        )

    else:

        message = ""

    if flags & FLAG_IMAGE:

        if (
            width not in (8, 16, 24)
            or
            height != width
            or
            image_len != width * height
        ):

            raise ValueError(
                "Recovered image metadata is invalid."
            )

    return (
        message,
        image_bytes,
        width,
        height,
        flags
    )


# ============================================================
# AES-256-GCM
# ============================================================

def derive_key(
    password,
    salt
):

    if not password:

        raise ValueError(
            "AES Secret Key cannot be empty."
        )

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=AES_KEY_BYTES,
        salt=salt,
        iterations=PBKDF2_ITERATIONS
    )

    return kdf.derive(
        password.encode("utf-8")
    )


def aes_encrypt(
    payload,
    password
):

    salt = os.urandom(
        SALT_BYTES
    )

    nonce = os.urandom(
        NONCE_BYTES
    )

    key = derive_key(
        password,
        salt
    )

    cipher_len = (
        len(payload)
        +
        AES_TAG_BYTES
    )

    header = (
        PACKET_MAGIC
        +
        bytes([PACKET_VERSION])
        +
        salt
        +
        nonce
        +
        struct.pack(
            ">I",
            cipher_len
        )
    )

    ciphertext = AESGCM(
        key
    ).encrypt(
        nonce,
        payload,
        header
    )

    return (
        header
        +
        ciphertext
    )


def aes_decrypt(
    packet,
    password
):

    try:

        if len(packet) < PACKET_HEADER_SIZE:

            return (
                None,
                "Incomplete encrypted packet."
            )

        offset = 0

        magic = packet[
            offset:
            offset + 4
        ]

        offset += 4

        if magic != PACKET_MAGIC:

            return (
                None,
                "Invalid Deep Cipher packet."
            )

        version = packet[offset]
        offset += 1

        if version != PACKET_VERSION:

            return (
                None,
                "Unsupported packet version."
            )

        salt = packet[
            offset:
            offset + SALT_BYTES
        ]

        offset += SALT_BYTES

        nonce = packet[
            offset:
            offset + NONCE_BYTES
        ]

        offset += NONCE_BYTES

        cipher_len = struct.unpack(
            ">I",
            packet[
                offset:
                offset + 4
            ]
        )[0]

        total_size = (
            PACKET_HEADER_SIZE
            +
            cipher_len
        )

        if len(packet) < total_size:

            return (
                None,
                "Encrypted packet is incomplete."
            )

        header = packet[
            :PACKET_HEADER_SIZE
        ]

        ciphertext = packet[
            PACKET_HEADER_SIZE:
            total_size
        ]

        key = derive_key(
            password,
            salt
        )

        plaintext = AESGCM(
            key
        ).decrypt(
            nonce,
            ciphertext,
            header
        )

        return (
            plaintext,
            "AES-256-GCM authentication verified."
        )

    except InvalidTag:

        return (
            None,
            (
                "ACCESS DENIED: wrong key "
                "or hidden data was corrupted."
            )
        )

    except Exception as exc:

        return (
            None,
            f"AES error: {exc}"
        )


# ============================================================
# BIT HELPERS
# ============================================================

def packet_to_chunks(packet):

    chunks = []

    for start in range(
        0,
        len(packet),
        CHUNK_BYTES
    ):

        chunk = packet[
            start:
            start + CHUNK_BYTES
        ]

        chunk = chunk.ljust(
            CHUNK_BYTES,
            b"\x00"
        )

        chunks.append(
            chunk
        )

    return chunks


def chunk_to_bits(chunk):

    chunk = chunk.ljust(
        CHUNK_BYTES,
        b"\x00"
    )[:CHUNK_BYTES]

    bits = []

    for value in chunk:

        for position in range(
            7,
            -1,
            -1
        ):

            bits.append(
                (
                    value
                    >>
                    position
                )
                &
                1
            )

    return torch.tensor(
        bits,
        dtype=torch.float32
    ).unsqueeze(0)


def bits_to_chunk(bits):

    values = (
        bits.detach()
        .cpu()
        .flatten()
        .int()
        .tolist()
    )

    output = bytearray()

    for start in range(
        0,
        SECRET_BITS,
        8
    ):

        value = 0

        for bit in values[
            start:
            start + 8
        ]:

            value = (
                value << 1
            ) | int(bit)

        output.append(
            value
        )

    return bytes(output)


# ============================================================
# AUDIO HELPERS
# ============================================================

def load_cover_audio(filepath):

    if not filepath:

        raise ValueError(
            "Upload a Cover Audio file."
        )

    audio_np, _ = librosa.load(
        filepath,
        sr=SAMPLE_RATE,
        mono=True
    )

    audio_np = np.asarray(
        audio_np,
        dtype=np.float32
    )

    return torch.from_numpy(
        audio_np
    ).unsqueeze(0)


def normalize_segment(segment):

    segment = segment.float()

    peak = segment.abs().max()

    if peak > 1e-8:

        segment = (
            segment
            /
            peak
            *
            0.95
        )

    return segment


def calculate_snr(
    cover,
    stego
):

    cover = (
        cover.detach()
        .cpu()
        .float()
    )

    stego = (
        stego.detach()
        .cpu()
        .float()
    )

    signal_power = torch.mean(
        cover ** 2
    )

    noise_power = torch.mean(
        (
            cover
            -
            stego
        )
        **
        2
    )

    return float(
        (
            10
            *
            torch.log10(
                (
                    signal_power
                    +
                    1e-12
                )
                /
                (
                    noise_power
                    +
                    1e-12
                )
            )
        ).item()
    )


def encode_one_chunk(
    cover_segment,
    chunk
):

    model = require_model()

    target_bits = (
        chunk_to_bits(chunk)
        .to(DEVICE)
    )

    cover_segment = (
        normalize_segment(
            cover_segment
        )
        .to(DEVICE)
    )

    old_strength = float(
        model.embedding_strength
    )

    successes = []
    best_failure = None

    try:

        for strength in ADAPTIVE_STRENGTHS:

            model.embedding_strength = strength

            with torch.no_grad():

                stego_audio, logits = model(
                    cover_segment,
                    target_bits
                )

            recovered = (
                torch.sigmoid(logits)
                >=
                0.5
            ).float()

            errors = int(
                (
                    recovered
                    !=
                    target_bits
                )
                .sum()
                .item()
            )

            ber = (
                errors
                /
                SECRET_BITS
            )

            result = {
                "success":
                    errors == 0,

                "stego":
                    stego_audio
                    .detach()
                    .cpu(),

                "strength":
                    strength,

                "accuracy":
                    100.0
                    *
                    (
                        1.0
                        -
                        ber
                    ),

                "ber":
                    ber,

                "snr":
                    calculate_snr(
                        cover_segment,
                        stego_audio
                    ),
            }

            if errors == 0:

                successes.append(
                    result
                )

            if (
                best_failure is None
                or
                result["ber"]
                <
                best_failure["ber"]
                or
                (
                    result["ber"]
                    ==
                    best_failure["ber"]
                    and
                    result["snr"]
                    >
                    best_failure["snr"]
                )
            ):

                best_failure = result

        if successes:

            return max(
                successes,
                key=lambda result:
                    result["snr"]
            )

        return best_failure

    finally:

        model.embedding_strength = (
            old_strength
        )


def adaptive_embed_packet(
    cover_audio,
    packet
):

    chunks = packet_to_chunks(
        packet
    )

    required_chunks = len(
        chunks
    )

    total_samples = int(
        cover_audio.shape[-1]
    )

    original_duration = (
        total_samples
        /
        SAMPLE_RATE
    )

    available_segments = (
        total_samples
        //
        AUDIO_LENGTH
    )

    if available_segments < required_chunks:

        return (
            None,
            (
                "Cover audio is too short. "
                f"Required theoretical minimum: "
                f"{required_chunks * SEGMENT_SECONDS}s."
            )
        )

    successful_segments = []

    chunk_index = 0
    cover_index = 0

    best_accuracy = 0.0
    best_ber = 1.0

    while (
        chunk_index < required_chunks
        and
        cover_index < available_segments
    ):

        start = (
            cover_index
            *
            AUDIO_LENGTH
        )

        end = (
            start
            +
            AUDIO_LENGTH
        )

        segment = (
            cover_audio[
                :,
                start:end
            ]
            .unsqueeze(0)
        )

        result = encode_one_chunk(
            segment,
            chunks[chunk_index]
        )

        if result["success"]:

            successful_segments.append(
                result["stego"]
            )

            chunk_index += 1

            best_accuracy = 0.0
            best_ber = 1.0

        else:

            best_accuracy = max(
                best_accuracy,
                result["accuracy"]
            )

            best_ber = min(
                best_ber,
                result["ber"]
            )

        cover_index += 1

    if chunk_index < required_chunks:

        return (
            None,
            (
                "Embedding failed after retries. "
                f"Embedded {chunk_index}/{required_chunks} chunks. "
                f"Best accuracy for current chunk: "
                f"{best_accuracy:.2f}%, "
                f"BER {best_ber:.6f}. "
                "Try a longer or different cover audio."
            )
        )

    hidden_part = torch.cat(
        successful_segments,
        dim=2
    ).squeeze(0).cpu()

    hidden_samples = int(
        hidden_part.shape[-1]
    )

    final_stego = (
        cover_audio.detach()
        .cpu()
        .clone()
    )

    final_stego[
        :,
        :hidden_samples
    ] = hidden_part

    final_stego = final_stego[
        :,
        :total_samples
    ]

    return (
        final_stego,
        {
            "required_chunks":
                required_chunks,

            "available_segments":
                available_segments,

            "tested_segments":
                cover_index,

            "skipped_segments":
                cover_index
                -
                required_chunks,

            "original_duration":
                original_duration,

            "final_duration":
                final_stego.shape[-1]
                /
                SAMPLE_RATE,

            "hidden_duration":
                required_chunks
                *
                SEGMENT_SECONDS,
        }
    )


def load_stego_exact(filepath):

    audio, sr = sf.read(
        filepath,
        dtype="float32",
        always_2d=True
    )

    audio = audio.T

    if sr != SAMPLE_RATE:

        raise ValueError(
            "Stego WAV must be 16000 Hz."
        )

    if audio.shape[0] != 1:

        raise ValueError(
            "Stego WAV must be mono."
        )

    return torch.tensor(
        audio,
        dtype=torch.float32
    )


def decode_one_segment(segment):

    model = require_model()

    segment = segment.to(
        DEVICE
    )

    with torch.no_grad():

        logits = (
            model.secret_decoder(segment)
            .squeeze(1)
        )

        probabilities = torch.sigmoid(
            logits
        )

        bits = (
            probabilities
            >=
            0.5
        ).float()

        confidence = (
            torch.where(
                bits > 0.5,
                probabilities,
                1.0 - probabilities
            )
            .mean()
            .item()
            *
            100.0
        )

    return bits, confidence


# ============================================================
# SENDER / RECEIVER FUNCTIONS
# ============================================================

def calculate_capacity(
    message,
    image_file,
    image_size_option
):

    message_bytes = len(
        (message or "").encode("utf-8")
    )

    if image_file:

        side = parse_image_size(
            image_size_option
        )

        image_bytes = (
            side * side
        )

    else:

        side = 0
        image_bytes = 0

    payload_bytes = (
        PAYLOAD_HEADER_SIZE
        +
        message_bytes
        +
        image_bytes
    )

    packet_bytes = (
        PACKET_HEADER_SIZE
        +
        payload_bytes
        +
        AES_TAG_BYTES
    )

    chunks = math.ceil(
        packet_bytes
        /
        CHUNK_BYTES
    )

    seconds = (
        chunks
        *
        SEGMENT_SECONDS
    )

    image_text = (
        f"{side}x{side}"
        if image_file
        else
        "None"
    )

    return (
        f"Message bytes: {message_bytes}\n"
        f"Image: {image_text} "
        f"({image_bytes} bytes)\n"
        f"Combined payload: {payload_bytes} bytes\n"
        f"AES packet: {packet_bytes} bytes\n"
        f"16-byte chunks: {chunks}\n"
        f"Theoretical minimum audio: "
        f"{seconds}s "
        f"({seconds // 60}m {seconds % 60}s)\n\n"
        "Use a longer cover audio because adaptive retry "
        "may skip difficult 3-second segments."
    )


def sender_process(
    cover_file,
    message,
    image_file,
    image_size_option,
    secret_key
):

    try:

        require_model()

        if not cover_file:

            return (
                None,
                None,
                "Upload a Cover Audio file."
            )

        if not secret_key:

            return (
                None,
                None,
                "Enter an AES Secret Key."
            )

        (
            payload,
            tiny_image,
            message_bytes,
            image_bytes,
            width,
            height
        ) = build_payload(
            message,
            image_file,
            image_size_option
        )

        packet = aes_encrypt(
            payload,
            secret_key
        )

        cover_audio = load_cover_audio(
            cover_file
        )

        final_stego, result = (
            adaptive_embed_packet(
                cover_audio,
                packet
            )
        )

        if final_stego is None:

            return (
                None,
                None,
                result
            )

        stamp = (
            time.strftime(
                "%Y%m%d_%H%M%S"
            )
            +
            "_"
            +
            secrets.token_hex(3)
        )

        preview_path = None

        if tiny_image is not None:

            preview_path = str(
                OUTPUT_DIR
                /
                f"sender_preview_{stamp}.png"
            )

            tiny_image.resize(
                (
                    width * 20,
                    height * 20
                ),
                Image.Resampling.NEAREST
            ).save(
                preview_path
            )

        output_path = str(
            OUTPUT_DIR
            /
            f"deepcipher_stego_{stamp}.wav"
        )

        sf.write(
            output_path,
            final_stego
            .squeeze(0)
            .numpy()
            .astype(np.float32),
            SAMPLE_RATE,
            subtype="FLOAT"
        )

        image_desc = (
            f"{width}x{height}, "
            f"{image_bytes} bytes"
            if image_bytes
            else
            "None"
        )

        status = (
            "MESSAGE + IMAGE EMBEDDING SUCCESS\n\n"
            f"Message bytes: {message_bytes}\n"
            f"Image: {image_desc}\n"
            f"AES packet: {len(packet)} bytes\n"
            f"AES chunks: "
            f"{result['required_chunks']}\n"
            f"Cover segments tested: "
            f"{result['tested_segments']}/"
            f"{result['available_segments']}\n"
            f"Skipped cover segments: "
            f"{result['skipped_segments']}\n"
            f"Hidden-data portion: "
            f"{result['hidden_duration']:.2f}s\n"
            f"Original audio duration: "
            f"{result['original_duration']:.2f}s\n"
            f"Final output duration: "
            f"{result['final_duration']:.2f}s\n\n"
            "Original duration preserved. "
            "Send the generated WAV to the receiver "
            "without converting or editing it."
        )

        return (
            output_path,
            preview_path,
            status
        )

    except Exception as exc:

        return (
            None,
            None,
            f"SENDER ERROR: {exc}"
        )


def receiver_process(
    stego_file,
    secret_key
):

    try:

        require_model()

        if not stego_file:

            return (
                "",
                None,
                None,
                "Upload the generated Stego WAV."
            )

        if not secret_key:

            return (
                "",
                None,
                None,
                "Enter the AES Secret Key."
            )

        stego_audio = load_stego_exact(
            stego_file
        )

        available_segments = (
            stego_audio.shape[-1]
            //
            AUDIO_LENGTH
        )

        total_audio_duration = (
            stego_audio.shape[-1]
            /
            SAMPLE_RATE
        )

        recovered_bytes = bytearray()
        confidences = []

        required_chunks = None
        total_packet_size = None

        for segment_index in range(
            available_segments
        ):

            start = (
                segment_index
                *
                AUDIO_LENGTH
            )

            end = (
                start
                +
                AUDIO_LENGTH
            )

            segment = (
                stego_audio[
                    :,
                    start:end
                ]
                .unsqueeze(0)
            )

            bits, confidence = (
                decode_one_segment(
                    segment
                )
            )

            recovered_bytes.extend(
                bits_to_chunk(
                    bits[0]
                )
            )

            confidences.append(
                confidence
            )

            if (
                required_chunks is None
                and
                len(recovered_bytes)
                >=
                PACKET_HEADER_SIZE
            ):

                header = bytes(
                    recovered_bytes[
                        :PACKET_HEADER_SIZE
                    ]
                )

                if header[:4] != PACKET_MAGIC:

                    return (
                        "",
                        None,
                        None,
                        (
                            "INVALID / CORRUPTED STEGO AUDIO: "
                            "packet header could not be recovered."
                        )
                    )

                if header[4] != PACKET_VERSION:

                    return (
                        "",
                        None,
                        None,
                        "Invalid packet version."
                    )

                cipher_position = (
                    4
                    +
                    1
                    +
                    SALT_BYTES
                    +
                    NONCE_BYTES
                )

                cipher_len = struct.unpack(
                    ">I",
                    header[
                        cipher_position:
                        cipher_position + 4
                    ]
                )[0]

                if (
                    cipher_len <= AES_TAG_BYTES
                    or
                    cipher_len > 10000
                ):

                    return (
                        "",
                        None,
                        None,
                        "Corrupted encrypted packet length."
                    )

                total_packet_size = (
                    PACKET_HEADER_SIZE
                    +
                    cipher_len
                )

                required_chunks = math.ceil(
                    total_packet_size
                    /
                    CHUNK_BYTES
                )

            if (
                required_chunks is not None
                and
                segment_index + 1
                >=
                required_chunks
            ):

                break

        if required_chunks is None:

            return (
                "",
                None,
                None,
                "AES packet header was not recovered."
            )

        if len(confidences) < required_chunks:

            return (
                "",
                None,
                None,
                "Incomplete hidden data."
            )

        packet = bytes(
            recovered_bytes[
                :total_packet_size
            ]
        )

        payload, aes_status = aes_decrypt(
            packet,
            secret_key
        )

        average_confidence = (
            sum(confidences)
            /
            len(confidences)
        )

        if payload is None:

            return (
                "",
                None,
                None,
                (
                    f"{aes_status}\n"
                    f"Average CNN confidence: "
                    f"{average_confidence:.2f}%"
                )
            )

        (
            message,
            image_bytes,
            width,
            height,
            flags
        ) = parse_payload(
            payload
        )

        recovered_preview = None
        recovered_file = None

        if (
            flags & FLAG_IMAGE
            and
            image_bytes
        ):

            array = np.frombuffer(
                image_bytes,
                dtype=np.uint8
            ).reshape(
                height,
                width
            )

            image = Image.fromarray(
                array,
                mode="L"
            )

            stamp = (
                time.strftime(
                    "%Y%m%d_%H%M%S"
                )
                +
                "_"
                +
                secrets.token_hex(3)
            )

            recovered_file = str(
                OUTPUT_DIR
                /
                f"recovered_{stamp}.png"
            )

            image.save(
                recovered_file
            )

            recovered_preview = str(
                OUTPUT_DIR
                /
                f"recovered_preview_{stamp}.png"
            )

            image.resize(
                (
                    width * 20,
                    height * 20
                ),
                Image.Resampling.NEAREST
            ).save(
                recovered_preview
            )

        status = (
            "MESSAGE + IMAGE RECOVERED SUCCESSFULLY\n\n"
            f"{aes_status}\n"
            f"Decoded hidden segments: "
            f"{required_chunks}\n"
            f"Full stego duration: "
            f"{total_audio_duration:.2f}s\n"
            f"Average CNN confidence: "
            f"{average_confidence:.2f}%\n"
        )

        if flags & FLAG_MESSAGE:

            status += (
                "Secret message recovered.\n"
            )

        if flags & FLAG_IMAGE:

            status += (
                f"Secret image recovered "
                f"({width}x{height}).\n"
            )

        return (
            message,
            recovered_preview,
            recovered_file,
            status
        )

    except Exception as exc:

        return (
            "",
            None,
            None,
            f"RECEIVER ERROR: {exc}"
        )


# ============================================================
# FASTAPI + SETUP PAGE
# ============================================================

app = FastAPI(
    title="Deep Cipher"
)


@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    ready = (
        load_model()
        is not None
    )

    state = (
        "READY"
        if ready
        else
        "MODEL SETUP REQUIRED"
    )

    return f"""
    <html>
    <head>
      <title>Deep Cipher</title>
    </head>
    <body style="
        font-family:Arial;
        max-width:760px;
        margin:60px auto;
        padding:0 20px;
    ">
      <h1>Deep Cipher</h1>
      <p><b>Status:</b> {state}</p>
      <p><a href="/sender">Open Sender</a></p>
      <p><a href="/receiver">Open Receiver</a></p>
      <p><a href="/setup">Model Setup</a></p>
    </body>
    </html>
    """


@app.get("/health")
def health():

    return {
        "status":
            "ok",

        "model_ready":
            load_model()
            is not None
    }


@app.get(
    "/setup",
    response_class=HTMLResponse
)
def setup_page():

    ready = (
        load_model()
        is not None
    )

    if ready:

        state = (
            "Model is loaded and ready."
        )

    else:

        state = (
            _model_error
            or
            "Model not ready."
        )

    return f"""
    <html>
    <head>
      <title>Deep Cipher Setup</title>
    </head>
    <body style="
        font-family:Arial;
        max-width:760px;
        margin:60px auto;
        padding:0 20px;
    ">
      <h2>Deep Cipher Model Setup</h2>
      <p>{state}</p>
      <p>
        Upload your trained
        <code>deep_cipher_final.pth</code>
        checkpoint once.
      </p>

      <form
        action="/setup"
        method="post"
        enctype="multipart/form-data"
      >
        <p>
          <input
            type="password"
            name="token"
            placeholder="Setup token"
            required
          >
        </p>

        <p>
          <input
            type="file"
            name="model_file"
            accept=".pth"
            required
          >
        </p>

        <button type="submit">
          Upload Model
        </button>
      </form>

      <p><a href="/">Back</a></p>
    </body>
    </html>
    """


@app.post(
    "/setup",
    response_class=HTMLResponse
)
async def setup_upload(
    token: str = Form(...),
    model_file: UploadFile = File(...)
):

    if (
        not SETUP_TOKEN
        or
        not secrets.compare_digest(
            token,
            SETUP_TOKEN
        )
    ):

        return HTMLResponse(
            "<h3>Access denied: invalid setup token.</h3>",
            status_code=403
        )

    if not model_file.filename.lower().endswith(
        ".pth"
    ):

        return HTMLResponse(
            "<h3>Only .pth files are accepted.</h3>",
            status_code=400
        )

    data = await model_file.read()

    if (
        len(data) < 1000
        or
        len(data)
        >
        50 * 1024 * 1024
    ):

        return HTMLResponse(
            "<h3>Unexpected checkpoint file size.</h3>",
            status_code=400
        )

    temp_path = MODEL_PATH.with_suffix(
        ".tmp"
    )

    temp_path.write_bytes(
        data
    )

    temp_path.replace(
        MODEL_PATH
    )

    if load_model(force=True) is None:

        return HTMLResponse(
            (
                "<h3>Checkpoint saved but failed validation.</h3>"
                f"<pre>{_model_error}</pre>"
            ),
            status_code=400
        )

    return HTMLResponse(
        """
        <h3>Model uploaded and loaded successfully.</h3>
        <p>
          <a href="/sender">Open Sender</a>
          |
          <a href="/receiver">Open Receiver</a>
        </p>
        """
    )


# ============================================================
# SENDER WEB APP
# ============================================================

with gr.Blocks(
    title="Deep Cipher Sender"
) as sender_demo:

    gr.Markdown(
        """
# 📤 Deep Cipher Sender
### Hide a secret message + image inside audio

The output is a **full-length Float32 WAV**.
Send that exact WAV file to the receiver.
"""
    )

    cover_input = gr.File(
        label="Cover Audio",
        file_types=[
            ".wav",
            ".mp3",
            ".flac",
            ".ogg"
        ],
        type="filepath"
    )

    message_input = gr.Textbox(
        label="Secret Message",
        lines=6,
        placeholder="Enter secret message..."
    )

    image_input = gr.Image(
        label="Secret Image (optional)",
        type="filepath"
    )

    image_size = gr.Dropdown(
        choices=[
            "8x8 - Small / Fast",
            "16x16 - Recommended",
            "24x24 - Higher Detail / Longer Audio"
        ],
        value="16x16 - Recommended",
        label="Hidden Image Resolution"
    )

    sender_key = gr.Textbox(
        label="AES Secret Key",
        type="password",
        placeholder="Enter secret key..."
    )

    capacity_button = gr.Button(
        "📏 Calculate Required Audio"
    )

    capacity_output = gr.Textbox(
        label="Capacity Information",
        lines=8
    )

    send_button = gr.Button(
        "🔐 Encrypt & Hide"
    )

    stego_output = gr.File(
        label="Generated Full-Length Stego WAV"
    )

    hidden_image_preview = gr.Image(
        label="Image Actually Hidden"
    )

    sender_status = gr.Textbox(
        label="Sender Status",
        lines=12
    )

    capacity_button.click(
        calculate_capacity,
        [
            message_input,
            image_input,
            image_size
        ],
        capacity_output
    )

    send_button.click(
        sender_process,
        [
            cover_input,
            message_input,
            image_input,
            image_size,
            sender_key
        ],
        [
            stego_output,
            hidden_image_preview,
            sender_status
        ]
    )


# ============================================================
# RECEIVER WEB APP
# ============================================================

with gr.Blocks(
    title="Deep Cipher Receiver"
) as receiver_demo:

    gr.Markdown(
        """
# 📥 Deep Cipher Receiver
### Recover the hidden message + image

Upload the **exact generated Stego WAV**.

Do not convert, normalize, or edit it before decoding.
"""
    )

    receiver_stego = gr.File(
        label="Stego WAV",
        file_types=[
            ".wav"
        ],
        type="filepath"
    )

    receiver_key = gr.Textbox(
        label="AES Secret Key",
        type="password",
        placeholder="Enter the same key..."
    )

    receive_button = gr.Button(
        "🔓 Extract & Decrypt"
    )

    recovered_message = gr.Textbox(
        label="Recovered Secret Message",
        lines=6
    )

    recovered_image = gr.Image(
        label="Recovered Secret Image"
    )

    recovered_png = gr.File(
        label="Recovered PNG File"
    )

    receiver_status = gr.Textbox(
        label="Receiver Status",
        lines=10
    )

    receive_button.click(
        receiver_process,
        [
            receiver_stego,
            receiver_key
        ],
        [
            recovered_message,
            recovered_image,
            recovered_png,
            receiver_status
        ]
    )


app = gr.mount_gradio_app(
    app,
    sender_demo,
    path="/sender"
)

app = gr.mount_gradio_app(
    app,
    receiver_demo,
    path="/receiver"
)


# Do not crash startup if the model has not been uploaded yet.
load_model()
