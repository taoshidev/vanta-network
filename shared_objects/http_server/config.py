"""argparse flags for the signed-request HTTP server, replacing bt.Axon.add_args.

Port default (8091) matches bt.Axon's historical default so no port reassignment is
needed on upgrade — it doesn't collide with vanta_api's REST (48888), WS (8765), or the
RPC service range (50000-50027).
"""
import argparse


def add_http_server_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--http.port", type=int, default=8091, dest="http_port",
                         help="Port for the signed-request HTTP server (replaces --axon.port).")
    parser.add_argument("--http.external_port", type=int, default=None, dest="http_external_port",
                         help="External port to publish on-chain if different from --http.port "
                              "(replaces --axon.external_port).")
    parser.add_argument("--http.host", type=str, default="0.0.0.0", dest="http_host",
                         help="Bind host for the signed-request HTTP server.")
    parser.add_argument("--legacy-auth-enabled", action="store_true", default=True, dest="legacy_auth_enabled",
                         help="Accept requests signed with the legacy bittensor v10 Dendrite header "
                              "scheme, for peers that haven't upgraded yet.")
    parser.add_argument("--no-legacy-auth", action="store_false", dest="legacy_auth_enabled",
                         help="Reject legacy bittensor v10-signed requests outright "
                              "(flip on once no v10 peers are observed).")
