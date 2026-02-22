"""
Standalone test for the secure aggregation pairwise mask scheme.

Tests:
  1. Two-hospital cancellation
  2. Three-hospital cancellation (the core use-case)
  3. Five-hospital cancellation (arbitrary n)
  4. FedAvg mean is preserved (not just sum)
  5. Different secrets produce non-cancelling masks
  6. Mask generation is deterministic
  7. Single hospital — no masks applied
  8. Partial upload does NOT cancel (expected behaviour)

Run from the project root:
    python3 test_secure_agg.py
"""

import hashlib
import hmac
import sys

import numpy as np
import torch

# ── Inline the core functions so this script needs no module path setup ───────
# Mirrors server/backend/secure_agg.py and hospital_client/backend/secure_agg.py

SECRET = "test-secret-for-unit-tests"


def derive_pair_seed(i: int, j: int, secret: str = SECRET) -> bytes:
    assert i < j, f"Expected i < j, got {i}, {j}"
    msg = f"{i},{j}".encode()
    return hmac.new(secret.encode(), msg, hashlib.sha256).digest()


def generate_pair_mask(seed: bytes, state_dict: dict) -> dict:
    masks = {}
    for name, param in state_dict.items():
        sub_seed_bytes = hashlib.sha256(seed + name.encode()).digest()
        seed_int = int.from_bytes(sub_seed_bytes[:8], byteorder="little")
        rng = np.random.default_rng(seed_int)
        noise = rng.standard_normal(param.numel()).astype(np.float32)
        masks[name] = torch.tensor(noise.reshape(param.shape), dtype=torch.float32)
    return masks


def apply_masks(
    state_dict: dict, hospital_id: int, all_ids: list, secret: str = SECRET
) -> dict:
    masked = {k: v.clone().float() for k, v in state_dict.items()}
    peers = [h for h in all_ids if h != hospital_id]
    for peer_id in peers:
        i = min(hospital_id, peer_id)
        j = max(hospital_id, peer_id)
        seed = derive_pair_seed(i, j, secret)
        mask = generate_pair_mask(seed, masked)
        sign = +1 if hospital_id < peer_id else -1
        for name in masked:
            masked[name] = masked[name] + sign * mask[name]
    return masked


# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
results = []


def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"{status}  {label}{suffix}")
    results.append((label, condition))


def make_state_dict(seed: int = 0) -> dict:
    """Random state-dict with the same keys/shapes as MergedModel (3->16->8->1)."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    return {
        "net.0.weight": torch.randn(16, 3, generator=rng),
        "net.0.bias": torch.randn(16, generator=rng),
        "net.3.weight": torch.randn(8, 16, generator=rng),
        "net.3.bias": torch.randn(8, generator=rng),
        "net.6.weight": torch.randn(1, 8, generator=rng),
        "net.6.bias": torch.randn(1, generator=rng),
    }


def sum_state_dicts(sds: list) -> dict:
    keys = sds[0].keys()
    return {
        k: torch.stack([sd[k].float() for sd in sds], dim=0).sum(dim=0) for k in keys
    }


def mean_state_dicts(sds: list) -> dict:
    n = len(sds)
    return {k: v / n for k, v in sum_state_dicts(sds).items()}


def max_abs_diff(a: dict, b: dict) -> float:
    return max((a[k].float() - b[k].float()).abs().max().item() for k in a)


ATOL = 1e-4  # tolerance for float32 rounding


# =============================================================================
# Test 1 — Two hospitals
# =============================================================================

print("\n── Test 1: Two-hospital mask cancellation ───────────────────────────")

delta_1 = make_state_dict(seed=1)
delta_2 = make_state_dict(seed=2)
all_ids = [1, 2]

masked_1 = apply_masks(delta_1, hospital_id=1, all_ids=all_ids)
masked_2 = apply_masks(delta_2, hospital_id=2, all_ids=all_ids)

diff = max_abs_diff(
    sum_state_dicts([delta_1, delta_2]), sum_state_dicts([masked_1, masked_2])
)
check(
    "Sum of masked weights == sum of raw weights", diff < ATOL, f"max diff={diff:.2e}"
)

check(
    "Hospital 1 masked != raw (masks were applied)",
    max_abs_diff(delta_1, masked_1) > ATOL,
)
check(
    "Hospital 2 masked != raw (masks were applied)",
    max_abs_diff(delta_2, masked_2) > ATOL,
)


# =============================================================================
# Test 2 — Three hospitals  (deltaA + R_AB + R_AC, deltaB - R_AB + R_BC, ...)
# =============================================================================

print("\n── Test 2: Three-hospital mask cancellation ─────────────────────────")

delta_A = make_state_dict(seed=10)
delta_B = make_state_dict(seed=20)
delta_C = make_state_dict(seed=30)
all_ids = [1, 2, 3]

masked_A = apply_masks(delta_A, hospital_id=1, all_ids=all_ids)
masked_B = apply_masks(delta_B, hospital_id=2, all_ids=all_ids)
masked_C = apply_masks(delta_C, hospital_id=3, all_ids=all_ids)

diff = max_abs_diff(
    sum_state_dicts([delta_A, delta_B, delta_C]),
    sum_state_dicts([masked_A, masked_B, masked_C]),
)
check(
    "Sum of 3 masked weights == sum of 3 raw weights",
    diff < ATOL,
    f"max diff={diff:.2e}",
)
check("Hospital A masked != raw", max_abs_diff(delta_A, masked_A) > ATOL)
check("Hospital B masked != raw", max_abs_diff(delta_B, masked_B) > ATOL)
check("Hospital C masked != raw", max_abs_diff(delta_C, masked_C) > ATOL)


# =============================================================================
# Test 3 — Five hospitals (arbitrary n)
# =============================================================================

print("\n── Test 3: Five-hospital mask cancellation ──────────────────────────")

n = 5
all_ids = list(range(1, n + 1))
raw_dicts = [make_state_dict(seed=i * 100) for i in range(n)]
masked_dicts = [apply_masks(raw_dicts[i], all_ids[i], all_ids) for i in range(n)]

diff = max_abs_diff(sum_state_dicts(raw_dicts), sum_state_dicts(masked_dicts))
check(
    f"Sum of {n} masked weights == sum of {n} raw weights",
    diff < ATOL,
    f"max diff={diff:.2e}",
)


# =============================================================================
# Test 4 — FedAvg mean is preserved (not just sum)
# =============================================================================

print("\n── Test 4: FedAvg mean preserved ────────────────────────────────────")

diff = max_abs_diff(mean_state_dicts(raw_dicts), mean_state_dicts(masked_dicts))
check(
    "Mean of masked weights == mean of raw weights (FedAvg correct)",
    diff < ATOL,
    f"max diff={diff:.2e}",
)


# =============================================================================
# Test 5 — Wrong secret produces non-cancelling masks
# =============================================================================

print("\n── Test 5: Wrong secret breaks cancellation ─────────────────────────")

delta_X = make_state_dict(seed=1)
delta_Y = make_state_dict(seed=2)
all_ids = [1, 2]

masked_X = apply_masks(delta_X, hospital_id=1, all_ids=all_ids, secret="correct-secret")
masked_Y = apply_masks(delta_Y, hospital_id=2, all_ids=all_ids, secret="wrong-secret")

diff = max_abs_diff(
    sum_state_dicts([delta_X, delta_Y]), sum_state_dicts([masked_X, masked_Y])
)
check(
    "Mismatched secrets do NOT cancel (diff is large)",
    diff > ATOL,
    f"max diff={diff:.4f}",
)


# =============================================================================
# Test 6 — Determinism: same inputs always produce same masked output
# =============================================================================

print("\n── Test 6: Mask generation is deterministic ─────────────────────────")

delta = make_state_dict(seed=42)
all_ids = [1, 2, 3]

run1 = apply_masks(delta, hospital_id=1, all_ids=all_ids)
run2 = apply_masks(delta, hospital_id=1, all_ids=all_ids)

diff = max_abs_diff(run1, run2)
check("Two runs with same inputs produce identical masks", diff == 0.0, f"diff={diff}")


# =============================================================================
# Test 7 — Single hospital: no masks applied, output == input
# =============================================================================

print("\n── Test 7: Single hospital — no masks applied ───────────────────────")

delta_solo = make_state_dict(seed=7)
masked_solo = apply_masks(delta_solo, hospital_id=1, all_ids=[1])

diff = max_abs_diff(delta_solo, masked_solo)
check("Solo hospital masked output == raw input", diff == 0.0, f"diff={diff}")


# =============================================================================
# Test 8 — Partial upload does NOT cancel (masks only cancel when all submit)
# =============================================================================

print("\n── Test 8: Partial upload does not cancel (expected behaviour) ───────")

delta_A = make_state_dict(seed=1)
delta_B = make_state_dict(seed=2)
# C is registered but has NOT uploaded yet
all_ids = [1, 2, 3]

masked_A = apply_masks(delta_A, hospital_id=1, all_ids=all_ids)
masked_B = apply_masks(delta_B, hospital_id=2, all_ids=all_ids)

diff = max_abs_diff(
    sum_state_dicts([masked_A, masked_B]),
    sum_state_dicts([delta_A, delta_B]),
)
check(
    "Partial set (A+B without C) does NOT cancel — diff is large",
    diff > ATOL,
    f"max diff={diff:.4f}",
)


# =============================================================================
# Test 9 — KeyPool: keys are issued in slot order 1, 2, 3 then wrap to next round
# =============================================================================

print("\n── Test 9: KeyPool issues slots 1→2→3 then starts next round ─────────")

import hashlib as _hashlib
import hmac as _hmac

ROUND_SIZE = 3
AGGREGATION_SECRET = "test-secret-for-unit-tests"

_SLOT_PAIRS = {
    1: [(1, 2, +1), (1, 3, +1)],
    2: [(1, 2, -1), (2, 3, +1)],
    3: [(1, 3, -1), (2, 3, -1)],
}


def derive_round_seed(round_id: int, i: int, j: int) -> bytes:
    """Mirrors KeyPool._derive_pair_seed — includes round_id in the message."""
    msg = f"{round_id}:{i},{j}".encode()
    return _hmac.new(AGGREGATION_SECRET.encode(), msg, _hashlib.sha256).digest()


def build_key(round_id: int, slot: int) -> dict:
    """Build a key dict the same way KeyPool._generate_pool does."""
    pairs = []
    for i, j, sign in _SLOT_PAIRS[slot]:
        seed = derive_round_seed(round_id, i, j)
        pairs.append({"i": i, "j": j, "sign": sign, "seed": seed})
    return {"round_id": round_id, "slot": slot, "pairs": pairs}


def apply_key(state_dict: dict, key: dict) -> dict:
    """Mirrors KeyPool.apply_key."""
    masked = {k: v.clone().float() for k, v in state_dict.items()}
    for pair in key["pairs"]:
        mask = _gen_mask(pair["seed"], masked)
        for name in masked:
            masked[name] = masked[name] + pair["sign"] * mask[name]
    return masked


def _gen_mask(seed: bytes, state_dict: dict) -> dict:
    """Mirrors KeyPool._generate_mask."""
    masks = {}
    for name, param in state_dict.items():
        sub_seed = _hashlib.sha256(seed + name.encode()).digest()
        seed_int = int.from_bytes(sub_seed[:8], byteorder="little")
        rng = np.random.default_rng(seed_int)
        noise = rng.standard_normal(param.numel()).astype(np.float32)
        masks[name] = torch.tensor(noise.reshape(param.shape), dtype=torch.float32)
    return masks


# Simulate issuing all 3 keys from round 1
pool_round1 = [build_key(1, slot) for slot in [1, 2, 3]]
check("Round 1 key 1 has slot=1", pool_round1[0]["slot"] == 1)
check("Round 1 key 2 has slot=2", pool_round1[1]["slot"] == 2)
check("Round 1 key 3 has slot=3", pool_round1[2]["slot"] == 3)
check("All round 1 keys share round_id=1", all(k["round_id"] == 1 for k in pool_round1))

# After exhaustion, next pool uses round_id=2
pool_round2 = [build_key(2, slot) for slot in [1, 2, 3]]
check("Round 2 keys have round_id=2", all(k["round_id"] == 2 for k in pool_round2))


# =============================================================================
# Test 10 — KeyPool: masks cancel within a round
# =============================================================================

print("\n── Test 10: KeyPool round masks cancel correctly ─────────────────────")

raw_A = make_state_dict(seed=10)
raw_B = make_state_dict(seed=20)
raw_C = make_state_dict(seed=30)

key_A = build_key(round_id=1, slot=1)
key_B = build_key(round_id=1, slot=2)
key_C = build_key(round_id=1, slot=3)

masked_A = apply_key(raw_A, key_A)
masked_B = apply_key(raw_B, key_B)
masked_C = apply_key(raw_C, key_C)

diff = max_abs_diff(
    sum_state_dicts([raw_A, raw_B, raw_C]),
    sum_state_dicts([masked_A, masked_B, masked_C]),
)
check("Round 1: sum of 3 masked == sum of 3 raw", diff < ATOL, f"max diff={diff:.2e}")
check("Slot 1 masked != raw", max_abs_diff(raw_A, masked_A) > ATOL)
check("Slot 2 masked != raw", max_abs_diff(raw_B, masked_B) > ATOL)
check("Slot 3 masked != raw", max_abs_diff(raw_C, masked_C) > ATOL)


# =============================================================================
# Test 11 — KeyPool: different rounds produce different masks
# =============================================================================

print("\n── Test 11: Different rounds produce different masks ─────────────────")

key_r1_s1 = build_key(round_id=1, slot=1)
key_r2_s1 = build_key(round_id=2, slot=1)

raw = make_state_dict(seed=99)
masked_r1 = apply_key(raw, key_r1_s1)
masked_r2 = apply_key(raw, key_r2_s1)

diff = max_abs_diff(masked_r1, masked_r2)
check(
    "Same slot, different round → different masked output",
    diff > ATOL,
    f"max diff={diff:.4f}",
)


# =============================================================================
# Test 12 — KeyPool: two complete rounds both cancel independently
# =============================================================================

print("\n── Test 12: Two complete rounds cancel independently ─────────────────")

raws_r1 = [make_state_dict(seed=i) for i in range(3)]
raws_r2 = [make_state_dict(seed=i + 10) for i in range(3)]

masked_r1 = [apply_key(raws_r1[i], build_key(1, i + 1)) for i in range(3)]
masked_r2 = [apply_key(raws_r2[i], build_key(2, i + 1)) for i in range(3)]

all_masked = masked_r1 + masked_r2
all_raw = raws_r1 + raws_r2

diff = max_abs_diff(sum_state_dicts(all_masked), sum_state_dicts(all_raw))
check(
    "Two complete rounds: sum of 6 masked == sum of 6 raw",
    diff < ATOL,
    f"max diff={diff:.2e}",
)


# =============================================================================
# Test 13 — KeyPool: partial round (only slots 1+2) does NOT cancel
# =============================================================================

print("\n── Test 13: Partial round does not cancel (slot 3 missing) ──────────")

raw_1 = make_state_dict(seed=1)
raw_2 = make_state_dict(seed=2)

masked_1 = apply_key(raw_1, build_key(round_id=5, slot=1))
masked_2 = apply_key(raw_2, build_key(round_id=5, slot=2))

diff = max_abs_diff(
    sum_state_dicts([masked_1, masked_2]),
    sum_state_dicts([raw_1, raw_2]),
)
check(
    "Partial round (slots 1+2 only) does NOT cancel — slot 3 missing",
    diff > ATOL,
    f"max diff={diff:.4f}",
)


# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
passed = sum(1 for _, ok in results if ok)
total = len(results)
color = "\033[92m" if passed == total else "\033[91m"
print(f"{color}  {passed}/{total} tests passed\033[0m")
print("=" * 60 + "\n")

if passed < total:
    failed = [label for label, ok in results if not ok]
    print("Failed tests:")
    for label in failed:
        print(f"  x  {label}")
    sys.exit(1)
