"""
Guards that the non-idempotent, money-moving RPC client wrappers opt OUT of the self-heal's
at-least-once retry (retry=False), and that the wrappers which already guard themselves against a
retried lost-ACK (check-before-mutate, write-after-mutate-completes) are left on the auto-retrying
path instead.

_invoke_rpc re-executes a call on a transient error (server bounce / lost ACK). For methods with no
such guard that perform an irreversible on-chain slash/withdraw/deposit or mint a subaccount, a
silent re-execution double-applies real funds — these must pass retry=False so a lost ACK surfaces
to the caller instead of being re-fired. A regression either direction — sweeping an already-guarded
method into fail-fast, or reverting an unguarded one back to auto-retrying self._server.X — is worth
catching, so both are pinned here.
"""
import unittest
from unittest.mock import MagicMock, PropertyMock, patch

from vali_objects.contract.contract_client import ContractClient
from entity_management.entity_client import EntityClient
from vali_objects.utils.elimination.elimination_client import EliminationClient
from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient
from vali_objects.utils.entity_collateral.entity_collateral_client import EntityCollateralClient


class TestFinancialRpcFailFast(unittest.TestCase):

    @staticmethod
    def _client(cls):
        # Skip __init__ (no RPC connection); stub the one method the wrappers now call.
        c = object.__new__(cls)
        c._invoke_rpc = MagicMock(return_value=None)
        return c

    def _assert_fail_fast(self, mock, expected_method):
        call = mock.call_args
        self.assertEqual(call.args[0], expected_method)
        self.assertIs(call.kwargs.get("retry"), False,
                      f"{expected_method} must be invoked with retry=False (fail-fast)")

    def test_contract_money_movers_fail_fast(self):
        c = self._client(ContractClient)
        c.process_withdrawal_request(1.0, "ck", "hk")
        self._assert_fail_fast(c._invoke_rpc, "process_withdrawal_request_rpc")
        c.slash_miner_collateral("hk", 2.0)
        self._assert_fail_fast(c._invoke_rpc, "slash_miner_collateral_rpc")
        c.slash_miner_collateral_proportion("hk", 0.5)
        self._assert_fail_fast(c._invoke_rpc, "slash_miner_collateral_proportion_rpc")
        c.force_deposit(3.0, "hk")
        self._assert_fail_fast(c._invoke_rpc, "force_deposit_rpc")

    def test_entity_money_movers_fail_fast(self):
        c = self._client(EntityClient)
        c.create_subaccount("eh", 1000.0, "crypto")
        self._assert_fail_fast(c._invoke_rpc, "create_subaccount_rpc")

    def test_entity_already_guarded_methods_still_auto_retry(self):
        # register_entity and create_hl_subaccount already guard against a retried lost-ACK
        # (check-before-slash, write-after-slash-completes), so a re-execution safely no-ops.
        # They must NOT have been swept into the fail-fast change.
        c = self._client(EntityClient)
        with patch.object(EntityClient, "_server", new_callable=PropertyMock) as server_prop:
            server_mock = MagicMock()
            server_prop.return_value = server_mock
            c.register_entity("eh")
            c.create_hl_subaccount("eh", 1000.0, "0xabc")
        server_mock.register_entity_rpc.assert_called_once_with("eh")
        server_mock.create_hl_subaccount_rpc.assert_called_once()
        c._invoke_rpc.assert_not_called()

    def test_elimination_append_still_auto_retries(self):
        # append_elimination_row already guards against a retried lost-ACK (check-before-slash,
        # write-after-slash-completes), so it must NOT have been swept into the fail-fast change.
        c = self._client(EliminationClient)
        with patch.object(EliminationClient, "_server", new_callable=PropertyMock) as server_prop:
            server_mock = MagicMock()
            server_prop.return_value = server_mock
            c.append_elimination_row("hk", "SOME_REASON")
        server_mock.append_elimination_row_rpc.assert_called_once()
        c._invoke_rpc.assert_not_called()

    def test_promote_subaccount_fail_fast(self):
        c = self._client(ChallengePeriodClient)
        c.promote_subaccount("hk", 1_700_000_000_000, 250_000.0)
        self._assert_fail_fast(c._invoke_rpc, "promote_subaccount_rpc")

    def test_slash_on_realized_loss_fail_fast(self):
        c = self._client(EntityCollateralClient)
        c.slash_on_realized_loss("eh", "synth_hk", 100.0)
        self._assert_fail_fast(c._invoke_rpc, "slash_on_realized_loss_rpc")

    def test_safe_deposit_still_auto_retries(self):
        # process_deposit_request is nonce-idempotent (miner-signed) and SHOULD keep the self-heal;
        # it must NOT have been swept into the fail-fast change.
        c = self._client(ContractClient)
        with patch.object(ContractClient, "_server", new_callable=PropertyMock) as server_prop:
            server_mock = MagicMock()
            server_prop.return_value = server_mock
            c.process_deposit_request("0xdeadbeef")
        server_mock.process_deposit_request_rpc.assert_called_once_with("0xdeadbeef")
        c._invoke_rpc.assert_not_called()   # not swept into the fail-fast change


if __name__ == "__main__":
    unittest.main()
