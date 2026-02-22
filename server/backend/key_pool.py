"""
KeyPool — pre-computed round key pool for secure aggregation.

How it works
------------
The server maintains a pool of exactly ROUND_SIZE (3) pre-computed keys at all
times.  When an upload arrives, the server pops the next key, applies the
corresponding pairwise masks to the uploaded weights server-side, and stores the
masked result alongside the round_id and slot number.

When the pool is exhausted (all 3 keys issued), a new round is generated
immediately so keys are always ready — zero latency on upload.

Key structure
-------------
Each key describes one slot in a round:

    {
        "round_id": 5,
        "slot":     2,          # 1, 2, or 3
        "pairs": [
            {"i": 1, "j": 2, "sign": -1, "seed": <bytes>},
            {"i": 2, "j": 3, "sign": +1, "seed": <bytes>},
        ]
    }

Mask derivation
---------------
For each pair (i, j) in a round:

    seed_ij = HMAC-SHA256(AGGREGATION_SECRET, f"{round_id}:{i},{j}")

Including round_id in the message ensures masks are unique per round — the same
slot in round 5 has completely different masks than in round 6.

Per-parameter masks are then derived as:

    sub_seed = SHA256(seed_ij || param_name_bytes)
    rng      = PCG64(int.from_bytes(sub_seed[:8], "little"))
    mask     = rng.standard_normal(param.numel()).reshape(param.shape)

Sign convention
---------------
Slot 1 (lowest):  + R_12  + R_13
Slot 2 (middle):  - R_12  + R_23
Slot 3 (highest): - R_13  - R_23

Summing all three masked uploads:
    (Δ1 + R_12 + R_13)
  + (Δ2 - R_12 + R_23)
  + (Δ3 - R_13 - R_23)
  = Δ1 + Δ2 + Δ3          ← all R terms cancel exactly

Thread safety
-------------
issue_key() is protected by a threading.Lock so concurrent uploads each
receive a distinct key even under parallel Flask request handling.
"""

import hashlib
import hmac
import logging
import os
import threading

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

ROUND_SIZE: int = 3

AGGREGATION_SECRET: str = os.environ.get(
    "AGGREGATION_SECRET",
    "carp-default-secret-change-in-production",
)

# For ROUND_SIZE = 3, the three slots and the pairs each slot contributes to.
# Each entry is (i, j, sign) where i < j (canonical pair ordering).
#   sign = +1  →  this slot ADDS     R_ij
#   sign = -1  →  this slot SUBTRACTS R_ij
_SLOT_PAIRS: dict[int, list[tuple[int, int, int]]] = {
    1: [(1, 2, +1), (1, 3, +1)],
    2: [(1, 2, -1), (2, 3, +1)],
    3: [(1, 3, -1), (2, 3, -1)],
}


# ── KeyPool ───────────────────────────────────────────────────────────────────


class KeyPool:
    """
    Thread-safe pool of pre-computed round keys.

    Maintains exactly ROUND_SIZE ready-to-issue keys.  Issuing the last key
    of a round immediately triggers generation of the next round so the pool
    is never empty after initialisation.

    Parameters
    ----------
    initial_round_id : int
        The round ID to start from.  Pass ``max_round_id_in_db + 1`` on
        server startup to avoid reusing round IDs across restarts.
    """

    def __init__(self, initial_round_id: int = 1) -> None:
        self._lock = threading.Lock()
        self._round_id: int = initial_round_id
        self._pool: list[dict] = []
        self._generate_pool()

    # ── Seed helpers ──────────────────────────────────────────────────────────

    def _derive_pair_seed(self, round_id: int, i: int, j: int) -> bytes:
        """
        Derive a 32-byte seed for pair (i, j) in *round_id*.

        HMAC-SHA256(AGGREGATION_SECRET, "{round_id}:{i},{j}")
        """
        msg = f"{round_id}:{i},{j}".encode()
        return hmac.new(
            AGGREGATION_SECRET.encode(),
            msg,
            hashlib.sha256,
        ).digest()

    def _generate_mask(
        self,
        seed: bytes,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Generate one float32 noise tensor per parameter using *seed*.

        Each parameter gets an independent sub-seed:
            SHA256(pair_seed || param_name_bytes)
        """
        masks: dict[str, torch.Tensor] = {}
        for name, param in state_dict.items():
            sub_seed = hashlib.sha256(seed + name.encode()).digest()
            seed_int = int.from_bytes(sub_seed[:8], byteorder="little")
            rng = np.random.default_rng(seed_int)
            noise = rng.standard_normal(param.numel()).astype(np.float32)
            masks[name] = torch.tensor(
                noise.reshape(param.shape),
                dtype=torch.float32,
            )
        return masks

    # ── Pool management ───────────────────────────────────────────────────────

    def _generate_pool(self) -> None:
        """
        Pre-compute all ROUND_SIZE keys for self._round_id and store them
        in self._pool (ready to pop from the front).
        """
        self._pool = []
        for slot in range(1, ROUND_SIZE + 1):
            pairs = []
            for i, j, sign in _SLOT_PAIRS[slot]:
                seed = self._derive_pair_seed(self._round_id, i, j)
                pairs.append({"i": i, "j": j, "sign": sign, "seed": seed})
            self._pool.append(
                {
                    "round_id": self._round_id,
                    "slot": slot,
                    "pairs": pairs,
                }
            )
        logger.info(
            "KeyPool: pre-computed %d keys for round %d.",
            ROUND_SIZE,
            self._round_id,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def issue_key(self) -> dict:
        """
        Pop and return the next key from the pool.

        If issuing this key exhausts the pool, the next round is generated
        immediately so the pool is always replenished before returning.

        Returns
        -------
        dict with keys: round_id, slot, pairs
        """
        with self._lock:
            key = self._pool.pop(0)

            if not self._pool:
                # Pool just became empty — pre-generate next round now.
                self._round_id += 1
                self._generate_pool()

            logger.info(
                "KeyPool: issued round=%d slot=%d (%d key(s) remaining).",
                key["round_id"],
                key["slot"],
                len(self._pool),
            )
            return key

    def apply_key(
        self,
        state_dict: dict[str, torch.Tensor],
        key: dict,
    ) -> dict[str, torch.Tensor]:
        """
        Apply the pairwise masks described by *key* to *state_dict*.

        Returns a new masked state-dict; the input is not modified.

        Parameters
        ----------
        state_dict : raw (DP-trained) weight state-dict from the hospital
        key        : a key dict as returned by issue_key()
        """
        masked: dict[str, torch.Tensor] = {
            k: v.clone().float() for k, v in state_dict.items()
        }

        for pair in key["pairs"]:
            seed: bytes = pair["seed"]
            sign: int = pair["sign"]
            mask = self._generate_mask(seed, masked)
            for name in masked:
                masked[name] = masked[name] + sign * mask[name]

        logger.info(
            "KeyPool: applied mask for round=%d slot=%d (%d param(s)).",
            key["round_id"],
            key["slot"],
            len(masked),
        )
        return masked

    @property
    def current_round_id(self) -> int:
        """The round ID currently being filled (keys are being issued from it)."""
        with self._lock:
            # The pool contains the remaining keys of the round just generated.
            # The round_id of those keys is self._round_id.
            return self._round_id


# ── Singleton ─────────────────────────────────────────────────────────────────

_key_pool: KeyPool | None = None


def get_key_pool() -> KeyPool:
    """
    Return the global KeyPool singleton.

    Raises RuntimeError if init_key_pool() has not been called yet.
    """
    if _key_pool is None:
        raise RuntimeError(
            "KeyPool has not been initialised. "
            "Call init_key_pool() during server startup."
        )
    return _key_pool


def init_key_pool(initial_round_id: int = 1) -> KeyPool:
    """
    Initialise (or re-initialise) the global KeyPool singleton.

    Call this once during server startup, passing ``max_round_id + 1``
    from the database so round IDs are never reused across restarts.

    Parameters
    ----------
    initial_round_id : int
        Round to start from.  Defaults to 1 for a fresh deployment.

    Returns
    -------
    The newly created KeyPool instance.
    """
    global _key_pool
    _key_pool = KeyPool(initial_round_id=initial_round_id)
    logger.info(
        "KeyPool initialised: starting at round_id=%d.",
        initial_round_id,
    )
    return _key_pool
