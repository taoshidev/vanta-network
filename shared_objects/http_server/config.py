"""argparse flags for the signed-request HTTP server, replacing bt.Axon.add_args.

Port default (8091) matches bt.Axon's historical default so no port reassignment is
needed on upgrade — it doesn't collide with vanta_api's REST (48888), WS (8765), or the
RPC service range (50000-50027).
"""
import argparse


def add_http_server_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--http.port", type=int, default=8091, dest="http_port",
                         help="Port for the signed-request HTTP server (replaces --axon.port); "
                              "also the port published on-chain via ServeAxon.")
    parser.add_argument("--http.host", type=str, default="0.0.0.0", dest="http_host",
                         help="Bind host for the signed-request HTTP server.")
