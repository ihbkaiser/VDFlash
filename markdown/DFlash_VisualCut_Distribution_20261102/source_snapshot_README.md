# Run source snapshot

The four experiment/runner sources are copied from the workspace because their SHA256 values match the Colab run config. `src/train_Dflash_SpecForge/.../dflash.py` was downloaded from the Colab checkout and matches the run-config fingerprint. `runtime_bytecode/dflash.cpython-313.pyc` is the bytecode file collected from the Colab runtime for provenance review; it is not a model output. The runtime bytecode hash is recorded in `../artifact_audit.json`.
