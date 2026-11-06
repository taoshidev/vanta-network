import json
import requests
from typing import Dict, Any, Optional
from vanta_cli.src.config import VANTA_API_BASE_URL_TESTNET, VANTA_API_BASE_URL_MAINNET, VANTA_CLI_VERSION
from vanta_cli.src.utils.console import console

def make_api_request(
    endpoint: str,
    payload: Optional[Dict[str, Any]] = None,
    method: str = "POST",
    base_url: str = VANTA_API_BASE_URL_MAINNET,
) -> Optional[Dict[str, Any]]:
    """
    Make an API request to the Vanta backend.

    Args:
        endpoint: API endpoint (e.g., '/collateral/deposit', '/collateral/withdraw')
        payload: Request payload as dictionary (optional for GET requests)
        method: HTTP method (default: POST)
        base_url: Base URL for the API

    Returns:
        Response JSON as dictionary, or None if request failed
    """
    url = f"{base_url}{endpoint}"

    try:
        if payload is not None:
            # Inject Vanta CLI version into payload for tracking
            payload["version"] = VANTA_CLI_VERSION
            response = requests.request(method, url, json=payload)
        else:
            response = requests.request(method, url)

        try:
            return response.json()
        except json.JSONDecodeError:
            console.print(f"[red]❌ Invalid JSON response: {response.text}[/red]")
            return None

    except Exception as e:
        console.print(f"[red]❌ API request failed: {e}[/red]")
        return None
