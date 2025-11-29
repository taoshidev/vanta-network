# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# developer: Taoshidev
# Copyright © 2024 Taoshi Inc

import typing
import bittensor as bt
from pydantic import Field

from typing import List

class SendSignal(bt.Synapse):
    signal: typing.Dict = Field(default_factory=dict, title="Signal", frozen=False, max_length=4096)
    repo_version: str = Field("N/A", title="Repo version (use the same meta.json file as validator)", frozen=False, max_length=256)
    successfully_processed: bool = Field(False, title="Successfully Processed", frozen=False)
    error_message: str = Field("", title="Error Message", frozen=False, max_length=4096)
    validator_hotkey: str = Field("", title="Hotkey set by validator", frozen=False, max_length=256)
    order_json: str = Field("", title="New Order JSON set by validator", frozen=False)
    miner_order_uuid: str = Field("", title="Order UUID set by miner", frozen=False, max_length=256)
    computed_body_hash: str = Field("", title="Computed Body Hash", frozen=False)

SendSignal.required_hash_fields = ["signal"]

class GetPositions(bt.Synapse):
    positions: List[typing.Dict] = Field(default_factory=list, title="Positions", frozen=False)
    successfully_processed: bool = Field(False, title="Successfully Processed", frozen=False)
    error_message: str = Field("", title="Error Message", frozen=False)
    computed_body_hash: str = Field("", title="Computed Body Hash", frozen=False)
    version: int = Field(0, title="Version", frozen=False)

GetPositions.required_hash_fields = ["positions"]

class ValidatorCheckpoint(bt.Synapse):
    checkpoint: str = Field("", title="Checkpoint", frozen=False)
    successfully_processed: bool = Field(False, title="Successfully Processed", frozen=False)
    error_message: str = Field("", title="Error Message", frozen=False)
    validator_receive_hotkey: str = Field("", title="Hotkey set by receiving validator", frozen=False)
    computed_body_hash: str = Field("", title="Computed Body Hash", frozen=False)
ValidatorCheckpoint.required_hash_fields = ["checkpoint"]

class GetDashData(bt.Synapse):
    data: typing.Dict = Field(default_factory=dict, title="Dashboard Data", frozen=False)
    successfully_processed: bool = Field(False, title="Successfully Processed", frozen=False)
    error_message: str = Field("", title="Error Message", frozen=False)
    computed_body_hash: str = Field("", title="Computed Body Hash", frozen=False)
GetDashData.required_hash_fields = ["data"]

class CollateralRecord(bt.Synapse):
    collateral_record: typing.Dict = Field(default_factory=dict, title="Collateral Record", frozen=False, max_length=4096)
    successfully_processed: bool = Field(False, title="Successfully Processed", frozen=False)
    error_message: str = Field("", title="Error Message", frozen=False, max_length=4096)
    computed_body_hash: str = Field("", title="Computed Body Hash", frozen=False)
CollateralRecord.required_hash_fields = ["collateral_record"]

class AssetSelection(bt.Synapse):
    asset_selection: typing.Dict = Field(default_factory=dict, title="Asset Selection", frozen=False, max_length=4096)
    successfully_processed: bool = Field(False, title="Successfully Processed", frozen=False)
    error_message: str = Field("", title="Error Message", frozen=False, max_length=4096)
    computed_body_hash: str = Field("", title="Computed Body Hash", frozen=False)
AssetSelection.required_hash_fields = ["asset_selection"]
