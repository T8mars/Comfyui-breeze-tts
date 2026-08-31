# Changelog

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
