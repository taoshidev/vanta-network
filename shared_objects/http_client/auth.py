"""httpx.Auth adapter around bittensor.http_auth.sign — signs each request as it is sent.

Per SIGNED-REQUESTS.md: "Clients sign a fresh nonce for every attempt, including retries
of an identical body." Because httpx calls auth_flow once per actual send, a fresh
HttpAuthClient.aquery() call (or any new httpx request through this Auth) naturally gets
a fresh nonce — never hoist signing outside of a per-attempt request construction.
"""
from typing import Iterator

import bittensor as bt
import httpx


class HotkeyAuth(httpx.Auth):
    requires_request_body = True

    def __init__(self, wallet, receiver_ss58: str):
        self.wallet = wallet
        self.receiver_ss58 = receiver_ss58

    def auth_flow(self, request: httpx.Request) -> Iterator[httpx.Request]:
        request.headers.update(bt.http_auth.sign(
            self.wallet,
            method=request.method,
            path=request.url.raw_path.decode(),
            body=request.content,
            receiver_ss58=self.receiver_ss58,
        ))
        yield request
