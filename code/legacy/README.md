# Historical code: retained for provenance

These files preserve historical experiment logic, not supported release entry points. Their original locations and hashes are in `provenance/source_manifest.json`. In v2, comments/docstrings were translated into English and an unconditional execution guard was added before experiment code. Current hashes therefore differ from the original files. All scripts here fail immediately if executed or imported. Use the reviewed training entry in the root README.

Some retraining scripts overwrite existing predictions/checkpoints. The historical wind-selection function deletes and renames training output directories, and uses test-set results to select sources. The multi-seed inverter script chooses by test R². Preserve them when inspecting how the experiments evolved; use the active commands in the root README for new work.

The older scheduling entries use different routes/scenario construction from the selected manual first-frame matrix. They were not combined with that matrix or certified to reproduce its outputs.
