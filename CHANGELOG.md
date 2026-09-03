# Changelog

## 0.2.9 — 2026-09-03

- Adds the T8star-Aix GitHub, Hugging Face, Bilibili, and YouTube links to the package documentation.
- Keeps all eight node contracts, four frontend workflows, inline vocal events, and `transformers>=4.57,<6` compatibility unchanged.

## 0.2.8 — 2026-09-03

- Aligns the node package version with the desktop release that bundles Whisper Small and improves first-run batch/voice-library interactions.
- Node contracts and the `transformers>=4.57,<6` compatibility policy remain unchanged.

## 0.2.7 — 2026-09-03

- Aligns the node package with Voice Studio 0.2.7 and its complete reference-audio Voice Library 2.1 workflow.
- Keeps all eight node contracts, four frontend workflows, inline vocal events, and `transformers>=4.57,<6` compatibility unchanged.

## 0.2.6 — 2026-09-03

- Shows the canonical Chinese and English inline vocal-event syntax directly on every frontend workflow canvas.
- Adds vocal-event examples and tooltips to Design, Clone, Direction, and Voice Bundle text inputs.
- Adds regression coverage proving event markers are preserved rather than stripped or rewritten.
- Keeps the standalone desktop bundle and the install-only ComfyUI node package clearly separated.

## 0.2.5 — 2026-09-03

- Adds four real ComfyUI frontend workflows with top-level `nodes`, `links`, canvas positions, groups, widgets, preview, and save nodes.
- Keeps the existing `*_api.json` prompt examples, while clearly separating them from drag-and-drop `*_workflow.json` files.
- Adds automated graph/link validation so API-only examples cannot be mistaken for UI workflows again.
- Labels the license, reference-audio, transcript, and `.t8voice.zip` steps directly on the canvas.
- Serializes the current ComfyUI AUDIO outputs on both PreviewAudio and SaveAudio, and documents the expected missing-reference prompt.

## 0.2.4 — 2026-09-03

- Aligns the node bundle with Voice Studio 0.2.4 and its reliable per-line rerun, backfill, and automatic timeline remix workflow.
- Keeps all eight node contracts and the host-owned dependency policy unchanged; compatibility remains `transformers>=4.57,<6`.

## 0.2.3 — 2026-09-01

- Aligns the node bundle with Voice Studio 0.2.3 and its quick-start, performance-recipe, responsive navigation, and global task-feedback release.
- Keeps all eight node contracts and the host-owned Torch/Transformers policy unchanged; compatibility remains `transformers>=4.57,<6`.

## 0.2.2 — 2026-09-01

- Documents a `--no-deps` manual-install path and the exact `comfyui-breeze-tts-T8` folder name so protected ComfyUI packages stay host-owned.
- Validates Request, AUDIO and sampling-setting contracts before resuming the model on GPU; zero temperatures and repetition penalties are no longer accepted.
- Adds regression coverage for Transformers 4.57.x/5.x boundaries and both dependency manifests' protected-package exclusions.

## 0.2.1 — 2026-09-01

- Aligns the distributable node release with Voice Studio 0.2.1 and its hardened project, queue, and voice-bundle workflows.
- Retains eight composable nodes and the host-owned dependency policy for Transformers 4.57.x through 5.x.

## 0.2.0 — 2026-09-01

- Adds an offline `T8 desktop voice bundle` node for verified `.t8voice.zip` files exported by the desktop voice library.
- Converts embedded references directly to standard ComfyUI `AUDIO` without extracting archive data to disk.
- Validates archive paths, Windows case-fold collisions, member counts, sizes, compression ratios and SHA-256 hashes before decode.
- Adds a composable per-line `inherit` / `override` / `neutral` natural-language direction helper; clone overrides retain the reference and route through Direction mode.
- Keeps the host-owned Torch, Transformers, Tokenizers and NumPy policy; compatibility remains `transformers>=4.57,<6`, including tested 4.57.x and 5.x lines.

## 0.1.2 — 2026-09-01

- Validates all required model, tokenizer, codec, index, and indexed shard files before loading, with resumable-download repair guidance.
- Rejects reference audio longer than 60 seconds before codec/GPU encoding.
- Serializes unload operations with generation and limits clone-unload hooks to the Breeze bundle actually being unloaded.
- Adds dependency-free model-integrity tests plus generation/unload guard tests.

## 0.1.1 — 2026-08-31

- Replaces file-descriptor warning suppression with direct CUDA graph cleanup.
- Selects Transformers mask-helper arguments from supported version boundaries without dynamic imports.
- Resolves two informational Registry scanner false positives while preserving Transformers 4.57 and 5.x compatibility.

## 0.1.0 — 2026-08-31

- Initial Comfy Registry release under Publisher ID `t8star`.
- Six composable nodes for model loading, voice design, voice cloning, voice direction, generation settings, and standard ComfyUI `AUDIO` generation.
- Real generation compatibility verified with Transformers 4.57.3 and 5.16.1.
- Includes Design, Clone, and Direction example API workflows.
- Pins the official Breeze TTS 2 model revision and verifies downloaded model files.
