#!/usr/bin/env python3
"""Dummy training script that exercises the Crucible training contract.

Reads config from env vars, prints stdout patterns that Crucible's OutputParser recognizes.
No actual ML work — just simulates the output format for testing.
"""
import os
import time
import random
import math

# Read config from environment
iterations = int(os.environ.get("ITERATIONS", "100"))
max_wallclock = int(os.environ.get("MAX_WALLCLOCK_SECONDS", "30"))
val_every = int(os.environ.get("VAL_LOSS_EVERY", "20"))
model_family = os.environ.get("MODEL_FAMILY", "baseline")
run_id = os.environ.get("RUN_ID", "unknown")

print(f"[dummy_train] Starting: family={model_family} iters={iterations} wallclock={max_wallclock}s")
print(f"[dummy_train] run_id={run_id}")

start_time = time.time()

# Simulate warmup
warmup_steps = min(10, iterations // 5)
for step in range(1, warmup_steps + 1):
    print(f"warmup_step:{step}/{warmup_steps}")
    time.sleep(0.01)

# Simulate training
best_val_loss = 10.0
for step in range(1, iterations + 1):
    elapsed = time.time() - start_time
    if elapsed > max_wallclock:
        print(f"stopping_early...step:{step}/{iterations}")
        break

    # Simulated decreasing train loss
    train_loss = 3.0 * math.exp(-step / (iterations / 3)) + 0.5 + random.gauss(0, 0.02)
    print(f"step:{step}/{iterations} train_loss:{train_loss:.4f}")

    # Periodic validation
    if step % val_every == 0 or step == iterations:
        val_loss = train_loss + random.gauss(0.1, 0.02)
        val_bpb = val_loss * 0.8 + random.gauss(0, 0.01)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        print(f"step:{step}/{iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

    time.sleep(0.01)  # Don't spin too fast

# Final result
final_val_loss = best_val_loss
final_val_bpb = final_val_loss * 0.8
model_bytes = random.randint(8_000_000, 16_000_000)

print(f"train_time:{time.time() - start_time:.2f}s")
print(f"Serialized model to {model_bytes} bytes")
print(f"final_int8_zlib_roundtrip val_loss:{final_val_loss:.4f} val_bpb:{final_val_bpb:.4f}")
print(f"serialized_model_int8_zlib: {model_bytes} bytes")
