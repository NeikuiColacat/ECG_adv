# methods/ecgtwin_gen/

当前毕业设计主线只保留 ECGTwin textual-inversion prompt-token 方法。

## Active

- `prompt_token/`: center-class soft prompt token bank and trainer support.
- `scripts/ecgtwin_gen/train_center_prompt_tokens.py`: train center-class prompt tokens.
- `scripts/ecgtwin_gen/generate_center_prompt_token_synth.py`: generate synthetic ECG with prompt tokens.
- `scripts/ecgtwin_gen/gate_prompt_token_synth.py`: export gated synthetic pools.

The active token path writes target-center style into the ECGTwin text
conditioning path as 768-d prompt embeddings. Do not put target-center style
into `base_vector`.

## Historical

The old 256-d AdaLN/base-vector center-token hook and style-translator branch
are historical and live under `legacy/` in this archive branch:

```text
legacy/docs/pipelines/ecgtwin_center_prompt_token_pipeline.md
legacy/scripts/ecgtwin_gen/
```

Do not use those archived paths for the final thesis pipeline unless a new
experiment explicitly revives that historical route.
