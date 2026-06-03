from enum import Enum


class SignalOrigin(str, Enum):
    UI = "UI"
    API = "API"
    SDK = "SDK"
    TG_BOT = "TG_BOT"
    MINER = "MINER"
