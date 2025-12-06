propose a solution for a new feature "Entity miners"


 One miner hotkey VANTA_ENTITY_HOTKEY will correspond to an entity.

  We will track entities with an EntityManager which persists data to disk, offers getters and setters via a client,
  and has a server class that delegates to a manager instance (just like challenge_period flow).

  Each entity i,e VANTA_ENTITY_HOTKEY can have subaccounts (monotonically increasing id).
  Subaccounts get their own synthetic hotkey which is f"{VANTA_ENTITY_HOTKEY}_{subaccount_id}"
  If a subaccount gets eliminated, that id can never be assigned again. An entity can only have MAX_SUBACCOUNTS_PER_ENTITY
  subaccounts at once. The limit is 500. Thus instead of tracking eliminated subaccount ids,
  we can simply maintain the active subaccount ids as well as the next id to be assigned


  We must support rest api requests of entity data using an EntityManagerClient in rest server.

    1. `POST register_subaccount` → returns {success, subaccount_id, subaccount_uuid}
        1. Verifies entity collateral and slot allowance
    2. `GET subaccount_status/{subaccouunt_id}` → active/eliminated/unknown

This is the approach we want to utilize for the subaccount registration process:
VantaRestServer endpoint exposed which does the collateral operations (placeholder for now)
 and then returns the newly-registered subaccount id to the caller.
  The validator then send a synapse message to all other validators so they are synced with the new subaccount id.
  Refer for the flow in broadcast_asset_selection_to_validators to see how we should do this.


EntityManager (RPCServerBase) will have its own daemon that periodically assess elimination criteria for entitiy miners.
Put a placeholder in this logic for now.


Most endpoints in VantaRestServer will support subaccounts directly since the passed in hotkey can be synthetic and
our existing code will be able to work with synthetic hotkeys as long as we adjust the metagraph logic to detect
synthetic hotkeys (have an underscore) and then making the appropriate call to the EntityManagerClient to see if
that subaccount is still active. and if the VANTA_ENTITY_HOTKEY hotkey is still in the raw metagraph. Our has_hotkey method
with this update should allow to work smoothly but let me know if there are other key parts of our system that
need to be updated to support synthetic hotkeys.


1. The entity hotkey (VANTA_ENTITY_HOTKEY) cannot place orders itself. Only its subaccounts can. This will need
to be enforced in validator.py.

2. Account sizes for synthetic hotkeys is set to a fixed value using a ContractClient after a blackbox function
transfers collateral from VANTA_ENTITY_HOTKEY. Leave placeholder functions for this. This account size init is done during
the subaccount registration flow.

3. debt based scoring will read debt ledgers for all miners including subaccounts. It needs to agrgeagte the debt
ledgers for all subaccounts into a single debt ledger representing the sum of all subaccount performance.
The key for this debt ledger will simply be the entity hotkey (VANTA_ENTITY_HOTKEY).

4.  Sub-accounts challenge period is an instantaneous pass if they get 3% returns against 6% drawdown within 90 days. Just like how in mdd checker, we can get returns and drawdown in different intervals, we will implement this in our
EntityManager daemon. A PerfLedgerClient is thus needed.

- Each entity miner can host up to **500 active sub-accounts** 
