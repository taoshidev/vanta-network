# Copyright (c) 2025 Taoshi Inc
"""
PaymentLedger - Persistent ledger for USDC payment records.

Tracks all payout payments sent to subaccount payout addresses,
providing double-payment guards and audit history.

Persistence follows the same pattern as entities.json via ValiBkpUtils.
"""
import json
import os
import threading
import time
import uuid
from typing import Dict, List, Optional

import bittensor as bt
from pydantic import BaseModel, Field


class PaymentRecord(BaseModel):
    """A single USDC payment record."""
    payment_id: str = Field(description="Unique payment ID (UUID)")
    subaccount_uuid: str = Field(description="Subaccount UUID that was paid")
    synthetic_hotkey: str = Field(description="Synthetic hotkey of the subaccount")
    payout_address: str = Field(description="EVM address payment was sent to")
    amount_usd: float = Field(description="Payout amount in USD")
    amount_usdc_raw: int = Field(description="USDC amount in raw units (6 decimals)")
    tx_hash: Optional[str] = Field(default=None, description="On-chain transaction hash")
    status: str = Field(default="pending", description="Payment status: pending, confirmed, failed")
    payout_period_start_ms: int = Field(description="Start of payout period (ms)")
    payout_period_end_ms: int = Field(description="End of payout period (ms)")
    created_at_ms: int = Field(description="When the payment record was created (ms)")
    confirmed_at_ms: Optional[int] = Field(default=None, description="When the tx was confirmed (ms)")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    chain_id: int = Field(description="Chain ID (e.g. 8453 for Base)")
    block_number: Optional[int] = Field(default=None, description="Block number of confirmed tx")


class PayoutRunResult(BaseModel):
    """Summary of a single payout run."""
    run_id: str = Field(description="Unique run ID (UUID)")
    entity_hotkey: str = Field(description="Entity hotkey that initiated the run")
    total_subaccounts: int = Field(default=0, description="Total subaccounts considered")
    eligible_count: int = Field(default=0, description="Subaccounts eligible for payout")
    successful_count: int = Field(default=0, description="Payments successfully confirmed")
    failed_count: int = Field(default=0, description="Payments that failed")
    skipped_count: int = Field(default=0, description="Payments skipped (already paid or below min)")
    total_usd_paid: float = Field(default=0.0, description="Total USD paid out")
    total_gas_eth: float = Field(default=0.0, description="Total gas spent in ETH")
    period_start_ms: int = Field(description="Payout period start (ms)")
    period_end_ms: int = Field(description="Payout period end (ms)")
    payments: List[PaymentRecord] = Field(default_factory=list, description="Individual payment records")
    created_at_ms: int = Field(description="When this run started (ms)")


class PaymentLedger:
    """
    Thread-safe persistent ledger for USDC payment records.

    Persists to a JSON file using the same write pattern as entities.json.
    Provides double-payment guards via has_been_paid().
    """

    def __init__(self, ledger_file_path: str):
        """
        Args:
            ledger_file_path: Absolute path to the ledger JSON file.
        """
        self._file_path = ledger_file_path
        self._lock = threading.RLock()
        self._payments: Dict[str, PaymentRecord] = {}  # payment_id -> PaymentRecord
        self._load()

    def _load(self):
        """Load payment records from disk."""
        if not os.path.exists(self._file_path):
            return

        try:
            with open(self._file_path, "r") as f:
                data = json.load(f)

            for payment_id, record_dict in data.get("payments", {}).items():
                self._payments[payment_id] = PaymentRecord(**record_dict)

            bt.logging.info(f"[PAYMENT_LEDGER] Loaded {len(self._payments)} payment records from disk")
        except Exception as e:
            bt.logging.error(f"[PAYMENT_LEDGER] Error loading ledger from {self._file_path}: {e}")

    def _save(self):
        """Persist payment records to disk."""
        try:
            os.makedirs(os.path.dirname(self._file_path), exist_ok=True)
            data = {
                "payments": {
                    pid: record.model_dump() for pid, record in self._payments.items()
                }
            }
            with open(self._file_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            bt.logging.error(f"[PAYMENT_LEDGER] Error saving ledger to {self._file_path}: {e}")

    def add_payment(self, record: PaymentRecord) -> None:
        """Add a new payment record and persist."""
        with self._lock:
            self._payments[record.payment_id] = record
            self._save()

    def update_payment_status(
        self,
        payment_id: str,
        status: str,
        tx_hash: Optional[str] = None,
        block_number: Optional[int] = None,
        error_message: Optional[str] = None,
        confirmed_at_ms: Optional[int] = None
    ) -> bool:
        """
        Update the status of an existing payment record.

        Returns:
            True if payment was found and updated, False otherwise.
        """
        with self._lock:
            record = self._payments.get(payment_id)
            if not record:
                return False

            record.status = status
            if tx_hash is not None:
                record.tx_hash = tx_hash
            if block_number is not None:
                record.block_number = block_number
            if error_message is not None:
                record.error_message = error_message
            if confirmed_at_ms is not None:
                record.confirmed_at_ms = confirmed_at_ms

            self._save()
            return True

    def has_been_paid(self, subaccount_uuid: str, start_ms: int, end_ms: int) -> bool:
        """
        Check if a subaccount has already been paid for a given period.

        This is the double-payment guard. Returns True if there exists a confirmed
        payment for the same subaccount covering the same period.

        Args:
            subaccount_uuid: The subaccount UUID
            start_ms: Period start timestamp (ms)
            end_ms: Period end timestamp (ms)

        Returns:
            True if already paid, False otherwise.
        """
        with self._lock:
            for record in self._payments.values():
                if (
                    record.subaccount_uuid == subaccount_uuid
                    and record.payout_period_start_ms == start_ms
                    and record.payout_period_end_ms == end_ms
                    and record.status == "confirmed"
                ):
                    return True
            return False

    def get_pending_payments(self) -> List[PaymentRecord]:
        """Get all payments with pending status (for recovery on restart)."""
        with self._lock:
            return [r for r in self._payments.values() if r.status == "pending"]

    def get_payments_for_subaccount(self, subaccount_uuid: str) -> List[PaymentRecord]:
        """Get all payment records for a given subaccount."""
        with self._lock:
            return [r for r in self._payments.values() if r.subaccount_uuid == subaccount_uuid]

    def get_total_paid(self, subaccount_uuid: Optional[str] = None) -> float:
        """
        Get total USD amount paid out.

        Args:
            subaccount_uuid: If provided, only count payments for this subaccount.

        Returns:
            Total USD paid (confirmed payments only).
        """
        with self._lock:
            total = 0.0
            for record in self._payments.values():
                if record.status != "confirmed":
                    continue
                if subaccount_uuid and record.subaccount_uuid != subaccount_uuid:
                    continue
                total += record.amount_usd
            return total

    def get_all_payments(self) -> List[PaymentRecord]:
        """Get all payment records."""
        with self._lock:
            return list(self._payments.values())
