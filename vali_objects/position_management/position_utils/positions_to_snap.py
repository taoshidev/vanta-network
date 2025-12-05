import json

from vali_objects.vali_dataclasses.position import Position
from vali_objects.utils.vali_bkp_utils import CustomEncoder
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.price_fetcher.live_price_server import LivePriceFetcherServer

positions_to_snap = []

if __name__ == "__main__":
    secrets = ValiUtils.get_secrets()
    lpf = LivePriceFetcherServer(secrets, disable_ws=True)
    for i, position_json in enumerate(positions_to_snap):
        # build the positions as the order edits did not propagate to position-level attributes.
        pos = Position(**position_json)
        pos.rebuild_position_with_updated_orders(lpf)
        positions_to_snap[i] = pos.model_dump()

    for position_json in positions_to_snap:
        pos = Position(**position_json)
        pos.rebuild_position_with_updated_orders(lpf)
        assert pos.is_closed_position
        #print(pos.to_copyable_str())
        str_to_write = json.dumps(pos, cls=CustomEncoder)

        print(pos.model_dump_json(), '\n', str_to_write)


