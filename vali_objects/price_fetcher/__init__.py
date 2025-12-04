# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc

"""Price fetcher package - live price data fetching and management.

Note: Imports are lazy to avoid circular import issues.
Use explicit imports from submodules:
    from vali_objects.price_fetcher.live_price_fetcher import LivePriceFetcher
    from vali_objects.price_fetcher.live_price_client import LivePriceFetcherClient
    from vali_objects.price_fetcher.live_price_server import LivePriceFetcherServer
"""

def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == 'LivePriceFetcher':
        from vali_objects.price_fetcher.live_price_fetcher import LivePriceFetcher
        return LivePriceFetcher
    elif name == 'LivePriceFetcherClient':
        from vali_objects.price_fetcher.live_price_client import LivePriceFetcherClient
        return LivePriceFetcherClient
    elif name == 'LivePriceFetcherServer':
        from vali_objects.price_fetcher.live_price_server import LivePriceFetcherServer
        return LivePriceFetcherServer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['LivePriceFetcher', 'LivePriceFetcherClient', 'LivePriceFetcherServer']
