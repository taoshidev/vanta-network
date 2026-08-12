# The MIT License (MIT)
# Copyright (c) 2024 Yuma Rao
# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc

import typing
import uuid

import bittensor as bt
import pydantic
from pydantic import Field

from typing import List, Any, Optional
from shared_objects.log import logger

try:
    import bittensor.utils.networking
    _original_get_external_ip = bittensor.utils.networking.get_external_ip
    _external_ip = None

    def _get_external_ip() -> str:
        global _external_ip
        if _external_ip is None:
            _external_ip = _original_get_external_ip()
        return _external_ip

    bittensor.utils.networking.get_external_ip = _get_external_ip
except (AttributeError, ImportError):
    pass


class _DendriteInfo:
    """Minimal dendrite-compatible namespace for bt11 migration.

    In bt10 this was set by bt.Axon from the verified caller signature.
    In bt11 it is set by our Flask-based axon replacement after verifying
    the request with bt.http_auth.verify.
    """
    def __init__(self, hotkey: str = ""):
        self.hotkey = hotkey

    def __repr__(self):
        return f"_DendriteInfo(hotkey={self.hotkey!r})"


class Synapse(pydantic.BaseModel):
    """Local Synapse base class replacing bt.Synapse for bt11 compatibility.

    The `dendrite` field is a transient, non-serialized attribute that carries
    the verified caller hotkey.  It is set by the server after verifying the
    incoming request signature and must NOT be included in serialization or
    RPC transmission.
    """
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    # exclude=True keeps this field out of model_dump() / JSON serialization.
    # It IS preserved by pickle (Pydantic v2 __getstate__ includes __dict__).
    dendrite: Any = Field(default=None, exclude=True)

    def model_post_init(self, __context: Any) -> None:
        if self.dendrite is None:
            object.__setattr__(self, 'dendrite', _DendriteInfo())


class SendSignal(Synapse):
    signal: typing.Dict = Field(default_factory=dict, title="Signal", frozen=False, max_length=4096)
    repo_version: str = Field("N/A", title="Repo version (use the same meta.json file as validator)", frozen=False, max_length=256)
    successfully_processed: bool = Field(False, title="Successfully Processed", frozen=False)
    error_message: str = Field("", title="Error Message", frozen=False, max_length=4096)
    should_retry: bool = Field(True, title="Whether miner should retry this validator on failure", frozen=False)
    validator_hotkey: str = Field("", title="Hotkey set by validator", frozen=False, max_length=256)
    order_json: str = Field("", title="New Order JSON set by validator", frozen=False)
    miner_order_uuid: str = Field("", title="Order UUID set by miner", frozen=False, max_length=256)
    subaccount_id: typing.Optional[int] = Field(default=None, title="Subaccount ID for entity miners", frozen=False)
    computed_body_hash: str = Field("", title="Computed Body Hash", frozen=False)

    @staticmethod
    def parse_miner_uuid(synapse: "SendSignal"):
        temp = synapse.miner_order_uuid
        assert isinstance(temp, str), f"excepted string miner uuid but got {temp}"
        if not temp:
            logger.warning(f'miner_order_uuid is empty for miner_hotkey [{synapse.dendrite.hotkey}] miner_repo_version '
                               f'[{synapse.repo_version}]. Generating a new one.')
            temp = str(uuid.uuid4())
        return temp

SendSignal.required_hash_fields = ["signal"]

class GetPositions(Synapse):
    positions: List[typing.Dict] = Field(default_factory=list, title="Positions", frozen=False)
    successfully_processed: bool = Field(False, title="Successfully Processed", frozen=False)
    error_message: str = Field("", title="Error Message", frozen=False)
    computed_body_hash: str = Field("", title="Computed Body Hash", frozen=False)
    version: int = Field(0, title="Version", frozen=False)

GetPositions.required_hash_fields = ["positions"]

class ValidatorCheckpoint(Synapse):
    checkpoint: str = Field("", title="Checkpoint", frozen=False)
    successfully_processed: bool = Field(False, title="Successfully Processed", frozen=False)
    error_message: str = Field("", title="Error Message", frozen=False)
    validator_receive_hotkey: str = Field("", title="Hotkey set by receiving validator", frozen=False)
    computed_body_hash: str = Field("", title="Computed Body Hash", frozen=False)
ValidatorCheckpoint.required_hash_fields = ["checkpoint"]


class CollateralRecord(Synapse):
    collateral_record: typing.Dict = Field(default_factory=dict, title="Collateral Record", frozen=False, max_length=4096)
    successfully_processed: bool = Field(False, title="Successfully Processed", frozen=False)
    error_message: str = Field("", title="Error Message", frozen=False, max_length=4096)
    computed_body_hash: str = Field("", title="Computed Body Hash", frozen=False)
CollateralRecord.required_hash_fields = ["collateral_record"]

class AssetSelection(Synapse):
    asset_selection: typing.Dict = Field(default_factory=dict, title="Asset Selection", frozen=False, max_length=4096)
    successfully_processed: bool = Field(False, title="Successfully Processed", frozen=False)
    error_message: str = Field("", title="Error Message", frozen=False, max_length=4096)
    computed_body_hash: str = Field("", title="Computed Body Hash", frozen=False)
AssetSelection.required_hash_fields = ["asset_selection"]

class SubaccountRegistration(Synapse):
    subaccount_data: typing.Dict = Field(default_factory=dict, title="Subaccount Registration Data", frozen=False, max_length=4096)
    successfully_processed: bool = Field(False, title="Successfully Processed", frozen=False)
    error_message: str = Field("", title="Error Message", frozen=False, max_length=4096)
    computed_body_hash: str = Field("", title="Computed Body Hash", frozen=False)
SubaccountRegistration.required_hash_fields = ["subaccount_data"]

class EntityEndpointUpdate(Synapse):
    endpoint_data: typing.Dict = Field(default_factory=dict, title="Entity Endpoint Update Data", frozen=False, max_length=4096)
    successfully_processed: bool = Field(False, title="Successfully Processed", frozen=False)
    error_message: str = Field("", title="Error Message", frozen=False, max_length=4096)
    computed_body_hash: str = Field("", title="Computed Body Hash", frozen=False)
EntityEndpointUpdate.required_hash_fields = ["endpoint_data"]
