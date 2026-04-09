# Crucible Examples

Working projects in several modalities. Each example is a self-contained directory you can copy as a starting point.

## Index

| Directory | Modality | What it demonstrates |
|---|---|---|
| [`basic/`](basic/) | Minimal | Dummy trainer showing the Crucible output contract |
| [`parameter_golf/`](parameter_golf/) | Language model | Tied-embedding LM for the OpenAI Parameter Golf competition |
| [`diffusion/`](diffusion/) | Diffusion | DDPM UNet on MNIST with a custom data adapter |
| [`world_model/`](world_model/) | World model | JEPA-style latent world model on bouncing balls |
| [`huggingface_finetune/`](huggingface_finetune/) | Bring-your-own-trainer | DistilBERT fine-tune on SST-2 via HuggingFace Trainer |
| [`ssh_local/`](ssh_local/) | SSH-only | Runs on a single SSH-reachable machine (no RunPod account) |
| [`yolo_mcp_empty_dir/`](yolo_mcp_empty_dir/) | Agent-driven | How to run Ultralytics YOLO via MCP from an empty directory |
| [`workflows/`](workflows/) | Reference walkthroughs | 2-pod experiment, autonomous research, local smoke |

## Choosing a starting point

**New to Crucible?** Start with [`basic/`](basic/) to see the minimal shape of a Crucible-compatible training script, then [`diffusion/`](diffusion/) or [`world_model/`](world_model/) for a real example.

**Want to wrap an existing framework (HuggingFace, Lightning, FairSeq)?** See [`huggingface_finetune/`](huggingface_finetune/) — it shows how to read Crucible env vars in a script that uses an external training framework.

**Don't have a RunPod account yet?** See [`ssh_local/`](ssh_local/). Uses the SSH provider, so you can test the full fleet flow against a single box (localhost, a home server, or a remote VM).

**Testing LLM agent integration?** See [`yolo_mcp_empty_dir/`](yolo_mcp_empty_dir/) — walks Claude through running a YOLO training job via MCP tool use, starting from an empty directory.

## Running an example locally

Every example is a standalone Crucible project. To run one:

```bash
cd examples/diffusion
python train_generic.py                         # direct, no Crucible orchestration
# or
crucible run experiment --preset smoke           # via Crucible presets
```

## Running an example on a fleet

```bash
# From the example directory
crucible fleet provision --count 1
crucible fleet bootstrap
crucible run experiment --preset screen
```

Or use `crucible project new` to generate a project spec that points at your own fork:

```bash
crucible project new my-diffusion-fork --template diffusion \
    --set REPO_URL=https://github.com/me/my-diffusion-fork
```

## Contributing an example

Examples live under `examples/<modality>_<descriptor>/` and should contain:

- `README.md` — what it shows, how to run it, what the expected output looks like
- `crucible.yaml` — project-wide config (presets, provider, metrics)
- A training entry point — typically `train_generic.py` or `train.py`
- Any custom Python (model, data adapter, objective) the example needs
- Small artifacts only — if you need sample data, fetch it at runtime

If your example depends on non-core Crucible plugins, vendor them as a local `.crucible/plugins/<type>/<name>.py` alongside the example so it runs standalone.
