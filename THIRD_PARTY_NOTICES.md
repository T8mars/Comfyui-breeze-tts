# Third-party notices

This is an unofficial integration and is not endorsed by BreezeBlue or RESONIA, INC.

## Breeze TTS 2 source code

- Project: `breezeblue-ai/breeze-tts`
- Source revision synchronized by this distribution: `e2c5ac2f54fe15daa94237a7dbf31e446660a4c9`
- License: Apache License 2.0 (`LICENSE`)

## Breeze TTS 2 model materials

- Model: `BreezeBlue/Breeze-TTS-2`
- Pinned revision: `799624c0b4a1daa8db6d28bbd9850043c0270734`
- License: BreezeBlue Research and Non-Commercial License Agreement version 1.1 (`MODEL_LICENSE`)

Breeze TTS 2 is licensed under the BreezeBlue Research and Non-Commercial License Agreement. Copyright (c) 2026 RESONIA, INC. All Rights Reserved.

This node asks the local user to acknowledge the agreement before it downloads model materials. The open-weight model and self-hosted outputs remain non-commercial; BreezeBlue's paid hosted service has separate terms. Non-consensual voice cloning is prohibited.

## ComfyUI compatibility inference path

- Adapted from: `Saganaki22/ComfyUI-Breeze-TTS-2`
- Source revision: `0aaa6fd8a4694a9b504970ade4078be2d4620a0a`
- License: Apache License 2.0

The adapted files are `native.py`, `runtime.py`, `loader.py`, `int8.py`, and files under `vendor/`. T8 changes official-model resolution, fixed revisions, dependency checks, node composition, naming, and license acceptance.

## Qwen3-TTS codec implementation

Files under `vendor/codec_*` contain code adapted from `Qwen/Qwen3-TTS` / `qwen-tts 0.1.1`, licensed under Apache License 2.0. File headers retain provenance and describe compatibility changes.
