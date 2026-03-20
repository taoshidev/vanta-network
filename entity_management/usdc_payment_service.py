# Copyright (c) 2025 Taoshi Inc
"""
USDCPaymentService - Core logic for sending USDC payments to subaccount payout addresses.

Queries the validator API for payout amounts, then executes ERC-20 transfers on Base chain.
Uses PaymentLedger for double-payment prevention and audit trail.
"""
import json
import time
import uuid
from typing import Optional

import bittensor as bt
import requests as http_requests

from entity_management.payment_ledger import PaymentLedger, PaymentRecord, PayoutRunResult
from miner_config import MinerConfig

try:
    from web3 import Web3
    from web3.middleware import ExtraDataToPOAMiddleware
except ImportError:
    Web3 = None

# Minimal ERC-20 ABI for transfer and balanceOf
USDC_ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    }
]


class USDCPaymentService:
    """
    Handles USDC payment execution for entity miner subaccount payouts.

    Flow:
    1. Query validator for subaccount list and payout amounts
    2. Check double-payment guard in ledger
    3. Verify USDC and ETH balances
    4. Execute ERC-20 transfer() calls sequentially
    5. Wait for confirmation and update ledger
    """

    def __init__(
        self,
        private_key: str,
        rpc_url: str,
        validator_url: str,
        validator_api_key: str,
        payment_ledger: PaymentLedger,
        usdc_contract_address: str = MinerConfig.USDC_CONTRACT_ADDRESS_BASE,
        chain_id: int = MinerConfig.BASE_CHAIN_ID,
        hyperscaled_url: str = MinerConfig.HYPERSCALED_API_URL,
        slack_notifier=None
    ):
        if Web3 is None:
            raise ImportError("web3 is required for USDCPaymentService. Install with: pip install web3>=6.15.0")

        self._validator_url = validator_url.rstrip("/")
        self._validator_api_key = validator_api_key
        self._chain_id = chain_id
        self._ledger = payment_ledger
        self._slack_notifier = slack_notifier
        self._usdc_address = Web3.to_checksum_address(usdc_contract_address)
        self._hyperscaled_url = hyperscaled_url.rstrip("/")

        # Web3 setup
        self._w3 = Web3(Web3.HTTPProvider(rpc_url))
        self._w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        self._account = self._w3.eth.account.from_key(private_key)
        self._usdc_contract = self._w3.eth.contract(
            address=self._usdc_address,
            abi=USDC_ERC20_ABI
        )

        bt.logging.info(
            f"[USDC_PAYMENT] Initialized: wallet={self._account.address}, "
            f"chain_id={chain_id}, usdc={usdc_contract_address}"
        )

    def get_wallet_address(self) -> str:
        return self._account.address

    def get_usdc_balance(self) -> float:
        """Get USDC balance in human-readable units."""
        raw = self._usdc_contract.functions.balanceOf(self._account.address).call()
        return raw / (10 ** MinerConfig.USDC_DECIMALS)

    def get_eth_balance(self) -> float:
        """Get ETH balance for gas."""
        wei = self._w3.eth.get_balance(self._account.address)
        return self._w3.from_wei(wei, "ether")

    def process_payouts(
        self,
        entity_hotkey: str,
        start_time_ms: int,
        end_time_ms: int,
        dry_run: bool = False
    ) -> PayoutRunResult:
        """
        Execute the full payout flow for all eligible subaccounts.

        Args:
            entity_hotkey: Entity hotkey to process payouts for
            start_time_ms: Payout period start (ms)
            end_time_ms: Payout period end (ms)
            dry_run: If True, calculate but don't send transactions

        Returns:
            PayoutRunResult with summary and individual payment records
        """
        now_ms = int(time.time() * 1000)
        result = PayoutRunResult(
            run_id=str(uuid.uuid4()),
            entity_hotkey=entity_hotkey,
            period_start_ms=start_time_ms,
            period_end_ms=end_time_ms,
            created_at_ms=now_ms
        )

        # 1. Get subaccounts from validator
        subaccounts = self._query_entity_subaccounts(entity_hotkey)
        if subaccounts is None:
            bt.logging.error("[USDC_PAYMENT] Failed to query entity subaccounts from validator")
            return result

        result.total_subaccounts = len(subaccounts)

        # 2. Filter eligible subaccounts (has payout_address, active/funded/alpha status)
        eligible_statuses = {"active", "funded", "alpha"}
        eligible = []
        for sa in subaccounts:
            payout_address = sa.get("payout_address")
            status = sa.get("status", "")
            if not payout_address:
                continue
            if status not in eligible_statuses:
                continue
            eligible.append(sa)

        bt.logging.info(
            f"[USDC_PAYMENT] {len(eligible)}/{len(subaccounts)} subaccounts eligible for payout"
        )

        # 3. Query payout amounts and filter
        payout_queue = []  # (subaccount_dict, payout_usd)
        for sa in eligible:
            sub_uuid = sa.get("subaccount_uuid")
            payout_address = sa.get("payout_address")
            synthetic_hotkey = sa.get("synthetic_hotkey", "")

            # Double-payment guard
            if self._ledger.has_been_paid(sub_uuid, start_time_ms, end_time_ms):
                bt.logging.info(
                    f"[USDC_PAYMENT] Skipping {synthetic_hotkey}: already paid for this period"
                )
                result.skipped_count += 1
                continue

            # KYC verification — only pay accounts that have completed KYC
            if not self._check_kyc_status(payout_address):
                bt.logging.info(
                    f"[USDC_PAYMENT] Skipping {synthetic_hotkey}: KYC not approved for {payout_address}"
                )
                result.skipped_count += 1
                continue

            # Query validator for payout amount
            payout_usd = self._query_validator_payout(sub_uuid, start_time_ms, end_time_ms)
            if payout_usd is None:
                bt.logging.warning(f"[USDC_PAYMENT] Failed to get payout for {synthetic_hotkey}")
                result.skipped_count += 1
                continue

            if payout_usd < MinerConfig.PAYMENT_MIN_USDC_AMOUNT:
                bt.logging.info(
                    f"[USDC_PAYMENT] Skipping {synthetic_hotkey}: "
                    f"payout ${payout_usd:.2f} below minimum ${MinerConfig.PAYMENT_MIN_USDC_AMOUNT}"
                )
                result.skipped_count += 1
                continue

            payout_queue.append((sa, payout_usd))

        result.eligible_count = len(payout_queue)

        if not payout_queue:
            bt.logging.info("[USDC_PAYMENT] No eligible payouts to process")
            return result

        # 4. Check total USDC balance (abort entire run if insufficient)
        total_needed = sum(amount for _, amount in payout_queue)
        usdc_balance = self.get_usdc_balance()

        if usdc_balance < total_needed:
            msg = (
                f"[USDC_PAYMENT] Insufficient USDC balance: have ${usdc_balance:.2f}, "
                f"need ${total_needed:.2f} for {len(payout_queue)} payouts. Aborting entire run."
            )
            bt.logging.error(msg)
            if self._slack_notifier:
                self._slack_notifier.send_message(msg, level="error", bypass_cooldown=True)
            return result

        # 5. Check ETH balance for gas
        eth_balance = self.get_eth_balance()
        if eth_balance < 0.001:
            msg = f"[USDC_PAYMENT] ETH balance too low for gas: {eth_balance:.6f} ETH. Aborting."
            bt.logging.error(msg)
            if self._slack_notifier:
                self._slack_notifier.send_message(msg, level="error", bypass_cooldown=True)
            return result

        if dry_run:
            bt.logging.info(
                f"[USDC_PAYMENT] DRY RUN: Would pay {len(payout_queue)} subaccounts "
                f"totaling ${total_needed:.2f} USDC"
            )
            for sa, amount in payout_queue:
                bt.logging.info(
                    f"  {sa.get('synthetic_hotkey', 'unknown')}: "
                    f"${amount:.2f} -> {sa.get('payout_address')}"
                )
            result.eligible_count = len(payout_queue)
            return result

        # 6. Execute transfers sequentially
        for sa, payout_usd in payout_queue:
            sub_uuid = sa.get("subaccount_uuid")
            payout_address = sa.get("payout_address")
            synthetic_hotkey = sa.get("synthetic_hotkey", "")
            amount_raw = int(payout_usd * (10 ** MinerConfig.USDC_DECIMALS))

            # Create pending payment record
            payment = PaymentRecord(
                payment_id=str(uuid.uuid4()),
                subaccount_uuid=sub_uuid,
                synthetic_hotkey=synthetic_hotkey,
                payout_address=payout_address,
                amount_usd=payout_usd,
                amount_usdc_raw=amount_raw,
                payout_period_start_ms=start_time_ms,
                payout_period_end_ms=end_time_ms,
                created_at_ms=int(time.time() * 1000),
                chain_id=self._chain_id
            )
            self._ledger.add_payment(payment)

            # Send transfer with retry
            tx_hash = self._send_usdc_transfer_with_retry(payout_address, amount_raw)

            if tx_hash is None:
                self._ledger.update_payment_status(
                    payment.payment_id,
                    status="failed",
                    error_message="Transaction send failed after retries"
                )
                result.failed_count += 1
                result.payments.append(payment)
                continue

            # Wait for confirmation
            confirmed, block_num = self._wait_for_confirmation(tx_hash)

            if confirmed:
                self._ledger.update_payment_status(
                    payment.payment_id,
                    status="confirmed",
                    tx_hash=tx_hash,
                    block_number=block_num,
                    confirmed_at_ms=int(time.time() * 1000)
                )
                result.successful_count += 1
                result.total_usd_paid += payout_usd
                bt.logging.info(
                    f"[USDC_PAYMENT] Confirmed: {synthetic_hotkey} -> "
                    f"${payout_usd:.2f} USDC tx={tx_hash}"
                )
            else:
                self._ledger.update_payment_status(
                    payment.payment_id,
                    status="failed",
                    tx_hash=tx_hash,
                    error_message="Transaction not confirmed within timeout"
                )
                result.failed_count += 1
                bt.logging.error(
                    f"[USDC_PAYMENT] Failed confirmation: {synthetic_hotkey} tx={tx_hash}"
                )

            # Refresh payment record from ledger state
            result.payments.append(payment)

        # 7. Log summary
        bt.logging.info(
            f"[USDC_PAYMENT] Run complete: {result.successful_count} successful, "
            f"{result.failed_count} failed, {result.skipped_count} skipped, "
            f"${result.total_usd_paid:.2f} total paid"
        )

        if self._slack_notifier:
            self._slack_notifier.send_message(
                f"USDC Payout Run Complete\n"
                f"Entity: {entity_hotkey[:16]}...\n"
                f"Period: {start_time_ms} - {end_time_ms}\n"
                f"Successful: {result.successful_count}/{result.eligible_count}\n"
                f"Failed: {result.failed_count}\n"
                f"Skipped: {result.skipped_count}\n"
                f"Total Paid: ${result.total_usd_paid:.2f} USDC",
                level="success" if result.failed_count == 0 else "warning",
                bypass_cooldown=True
            )

        return result

    def _check_kyc_status(self, wallet_address: str) -> bool:
        """
        Check if a wallet has completed KYC via the Hyperscaled API.

        Args:
            wallet_address: EVM payout address to check

        Returns:
            True if KYC status is approved, False otherwise.
        """
        try:
            resp = http_requests.get(
                f"{self._hyperscaled_url}/api/kyc/status",
                params={"wallet": wallet_address},
                timeout=15
            )
            if resp.status_code != 200:
                bt.logging.warning(
                    f"[USDC_PAYMENT] KYC check failed for {wallet_address} "
                    f"({resp.status_code}): {resp.text}"
                )
                return False

            data = resp.json()
            verified = data.get("verified", False)
            kyc_status = data.get("kycStatus", "none")

            if not verified:
                bt.logging.info(
                    f"[USDC_PAYMENT] KYC not approved for {wallet_address} "
                    f"(status={kyc_status}), skipping payout"
                )
            return verified

        except Exception as e:
            bt.logging.error(f"[USDC_PAYMENT] Error checking KYC for {wallet_address}: {e}")
            return False

    def _query_entity_subaccounts(self, entity_hotkey: str) -> Optional[list]:
        """Query validator GET /entity/<entity_hotkey> for subaccount list."""
        try:
            resp = http_requests.get(
                f"{self._validator_url}/entity/{entity_hotkey}",
                headers={
                    "Authorization": f"Bearer {self._validator_api_key}",
                    "Content-Type": "application/json"
                },
                timeout=30
            )
            if resp.status_code != 200:
                bt.logging.error(
                    f"[USDC_PAYMENT] Validator entity query failed ({resp.status_code}): {resp.text}"
                )
                return None

            data = resp.json()
            entity_data = data.get("entity", {})

            # Extract subaccounts from entity data
            subaccounts_dict = entity_data.get("subaccounts", {})
            return list(subaccounts_dict.values())

        except Exception as e:
            bt.logging.error(f"[USDC_PAYMENT] Error querying entity subaccounts: {e}")
            return None

    def _query_validator_payout(
        self, subaccount_uuid: str, start_ms: int, end_ms: int
    ) -> Optional[float]:
        """Query validator POST /entity/subaccount/payout for payout amount."""
        try:
            resp = http_requests.post(
                f"{self._validator_url}/entity/subaccount/payout",
                json={
                    "subaccount_uuid": subaccount_uuid,
                    "start_time_ms": start_ms,
                    "end_time_ms": end_ms
                },
                headers={
                    "Authorization": f"Bearer {self._validator_api_key}",
                    "Content-Type": "application/json"
                },
                timeout=30
            )
            if resp.status_code != 200:
                bt.logging.warning(
                    f"[USDC_PAYMENT] Payout query failed for {subaccount_uuid} "
                    f"({resp.status_code}): {resp.text}"
                )
                return None

            data = resp.json()
            payout_data = data.get("data", {})
            return payout_data.get("payout")

        except Exception as e:
            bt.logging.error(f"[USDC_PAYMENT] Error querying payout for {subaccount_uuid}: {e}")
            return None

    def _send_usdc_transfer(self, to_address: str, amount_raw: int) -> Optional[str]:
        """
        Build, sign, and send a USDC transfer transaction.

        Args:
            to_address: Recipient EVM address
            amount_raw: USDC amount in raw units (6 decimals)

        Returns:
            Transaction hash hex string, or None on failure.
        """
        try:
            to_checksum = Web3.to_checksum_address(to_address)
            nonce = self._w3.eth.get_transaction_count(self._account.address)

            # Build transfer transaction
            tx = self._usdc_contract.functions.transfer(
                to_checksum, amount_raw
            ).build_transaction({
                "from": self._account.address,
                "nonce": nonce,
                "chainId": self._chain_id,
                "gas": 100_000,
                "maxFeePerGas": self._w3.eth.gas_price * 2,
                "maxPriorityFeePerGas": self._w3.to_wei(0.001, "gwei"),
            })

            # Estimate gas and apply buffer
            try:
                estimated_gas = self._w3.eth.estimate_gas(tx)
                tx["gas"] = int(estimated_gas * MinerConfig.PAYMENT_GAS_BUFFER_MULTIPLIER)
            except Exception:
                pass  # Use default 100k gas

            # Sign and send
            signed_tx = self._w3.eth.account.sign_transaction(tx, self._account.key)
            tx_hash = self._w3.eth.send_raw_transaction(signed_tx.raw_transaction)

            return tx_hash.hex()

        except Exception as e:
            bt.logging.error(f"[USDC_PAYMENT] Transfer failed to {to_address}: {e}")
            return None

    def _send_usdc_transfer_with_retry(self, to_address: str, amount_raw: int) -> Optional[str]:
        """Send USDC transfer with retry logic."""
        for attempt in range(MinerConfig.PAYMENT_MAX_RETRIES):
            tx_hash = self._send_usdc_transfer(to_address, amount_raw)
            if tx_hash:
                return tx_hash

            if attempt < MinerConfig.PAYMENT_MAX_RETRIES - 1:
                bt.logging.warning(
                    f"[USDC_PAYMENT] Transfer attempt {attempt + 1} failed, "
                    f"retrying in {MinerConfig.PAYMENT_RETRY_DELAY_S}s..."
                )
                time.sleep(MinerConfig.PAYMENT_RETRY_DELAY_S)

        return None

    def _wait_for_confirmation(
        self, tx_hash: str, timeout_s: int = MinerConfig.PAYMENT_CONFIRMATION_TIMEOUT_S
    ) -> tuple:
        """
        Wait for a transaction to be confirmed.

        Returns:
            (confirmed: bool, block_number: Optional[int])
        """
        try:
            receipt = self._w3.eth.wait_for_transaction_receipt(
                tx_hash, timeout=timeout_s
            )
            if receipt["status"] == 1:
                return True, receipt["blockNumber"]
            else:
                bt.logging.error(f"[USDC_PAYMENT] Tx reverted: {tx_hash}")
                return False, receipt.get("blockNumber")
        except Exception as e:
            bt.logging.error(f"[USDC_PAYMENT] Confirmation timeout/error for {tx_hash}: {e}")
            return False, None

    def check_pending_payments(self) -> int:
        """
        Check pending payments on-chain (recovery after restart).

        For each pending payment with a tx_hash, check if it was actually confirmed.

        Returns:
            Number of payments resolved.
        """
        pending = self._ledger.get_pending_payments()
        resolved = 0

        for payment in pending:
            if not payment.tx_hash:
                continue

            try:
                receipt = self._w3.eth.get_transaction_receipt(payment.tx_hash)
                if receipt:
                    if receipt["status"] == 1:
                        self._ledger.update_payment_status(
                            payment.payment_id,
                            status="confirmed",
                            block_number=receipt["blockNumber"],
                            confirmed_at_ms=int(time.time() * 1000)
                        )
                        bt.logging.info(
                            f"[USDC_PAYMENT] Recovered pending payment {payment.payment_id}: confirmed"
                        )
                    else:
                        self._ledger.update_payment_status(
                            payment.payment_id,
                            status="failed",
                            block_number=receipt["blockNumber"],
                            error_message="Transaction reverted (found during recovery)"
                        )
                        bt.logging.warning(
                            f"[USDC_PAYMENT] Recovered pending payment {payment.payment_id}: reverted"
                        )
                    resolved += 1
            except Exception as e:
                bt.logging.debug(
                    f"[USDC_PAYMENT] Could not check pending payment {payment.payment_id}: {e}"
                )

        if resolved:
            bt.logging.info(f"[USDC_PAYMENT] Resolved {resolved} pending payments from previous run")

        return resolved
