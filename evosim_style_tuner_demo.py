"""LUC-797 Phase 2 — style-tuner train + inference driver (run against local sst dev).

Two modes:

  train  — create a trivial Experiment (EvoSim kickoff requires one), write the
           training-run config, and call client.evosims.train(...). The rollouts hit
           the webhook harness; the run publishes a Checkpoint binding BOTH the
           style_tuner tool (artifact) AND the trained prompt version.

  infer  — bind a published checkpoint to a fresh session, fetch the target prompt
           (now the TRAINED version, via LUC-793/795 checkpoint-scoped resolution) and
           call the style_tuner tool (which returns the trained artifact).

The published checkpoint id is handed off manually (the SDK has no EvoSim-status
poll): after `train`, grab it from the dashboard (or a Checkpoint row with
source_type=EVOSIM for this agent) and pass it to `infer --checkpoint-id`.

Prereq: the target Prompt must already exist with a base `production` version — the
SDK cannot create prompts. Seed it once (see the runbook / LUC-797). Requires an SDK
build with LUC-795 for the checkpoint-scoped prompt fetch.

    LUCIDIC_API_KEY=... LUCIDIC_AGENT_ID=... LUCIDIC_BASE_URL=http://localhost:8000 \
      TARGET_PROMPT=style_tuner_demo_prompt python3 evosim_style_tuner_demo.py train
    ... python3 evosim_style_tuner_demo.py infer --checkpoint-id <uuid>
"""
import argparse
import json
import os
import tempfile

import lucidicai as lai

AGENT_ID = os.environ["LUCIDIC_AGENT_ID"]
TARGET_PROMPT = os.environ.get("TARGET_PROMPT", "style_tuner_demo_prompt")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "http://localhost:8799/")

client = lai.LucidicAI(providers=[])


def train() -> None:
    experiment_id = client.experiments.create(experiment_name="style-tuner demo")
    print(f"experiment_id = {experiment_id}")

    config = {
        "agent": AGENT_ID,
        "name": "style-tuner demo",
        "experiment_id": experiment_id,
        "webhook_url": WEBHOOK_URL,
        "module_selections": [
            {
                "module_key": "style_tuner",
                "target_prompt": TARGET_PROMPT,  # first-party edit-site field (LUC-799)
                "config": {
                    "session_specs": [{}, {}],  # 2 rollout sessions
                    "window": 2,
                    "dispatch_chunk": 2,
                    "deadline_seconds": 300,
                    "refine_iterations": 0,  # single pass; >0 exercises continue_as_new
                },
            }
        ],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(config, fh)
        cfg_path = fh.name
    result = client.evosims.train(cfg_path)
    print(f"train -> {json.dumps(result, indent=2, default=str)}")
    print(
        "\nMake sure the harness is running. When the run finishes, grab the published "
        "checkpoint id (dashboard / Checkpoint source_type=EVOSIM) and run:\n"
        "  python3 evosim_style_tuner_demo.py infer --checkpoint-id <uuid>"
    )


def infer(checkpoint_id: str) -> None:
    session = client.sessions.create(
        session_name="style-tuner inference", checkpoint_id=checkpoint_id
    )
    try:
        prompt = client.prompts.get(TARGET_PROMPT)  # -> TRAINED version via the checkpoint
        style = client.training_modules.call("get_style_guide", {})
        print("=== checkpoint-scoped prompt (trained) ===")
        print(prompt.content)
        print("\n=== style_tuner tool result (trained artifact) ===")
        print(json.dumps(style, indent=2, default=str))
    finally:
        session.end()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("train")
    p_infer = sub.add_parser("infer")
    p_infer.add_argument("--checkpoint-id", required=True)
    args = ap.parse_args()
    if args.mode == "train":
        train()
    else:
        infer(args.checkpoint_id)


if __name__ == "__main__":
    main()
