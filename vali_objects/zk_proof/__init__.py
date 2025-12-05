"""
ZK Proof Manager - Self-contained background worker for daily ZK proof generation.

This module provides ZKProofManager, a lightweight background thread that generates
ZK proofs for all active miners once per day. Results are saved to ~/.pop/ and
uploaded to api.omron.ai for external verification.

Architecture: Simple background thread pattern (no RPC) - similar to APIManager.
Not an RPC server because ZK proofs are for external verification only, not
consumed by validator operations.

Usage in validator.py:
    from vali_objects.zk_proof import ZKProofManager

    zk_manager = ZKProofManager(
        position_manager=self.position_manager_client,
        perf_ledger=self.perf_ledger_client,
        contract_manager=self.contract_client,
        wallet=self.wallet
    )
    zk_manager.start()
"""

from .zk_proof_manager import ZKProofManager

__all__ = ['ZKProofManager']
