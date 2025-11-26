# Miner

Our miners act like traders. To score well and receive incentive, they place **orders** on our system against different trade pairs. The magnitude of each order is determined by its **leverage**, which can be thought of as the percentage of the portfolio used for the transaction. An order with a leverage of 1.0x indicates that the miner is betting against their entire portfolio value.

The first time a miner places an order on a trade pair, they will open a **position** against it. The leverage and directionality of this position determines the miner's expectation of the trade pair's future movement. As long as this position is open, the miner is communicating an expectation of continued trade pair movement in this direction. There are two types of positions: **LONG** and **SHORT**.

A long position is a bet that the trade pair will increase, while a short position is a bet that the trade pair will decrease. Even if the overall position is LONG, a miner can submit a number of orders within this position to manage their risk exposure by adjusting the leverage. SHORT orders on a long position will reduce the overall leverage of the position, reducing the miner's exposure to the trade pair. LONG orders on a long position will increase the overall leverage of the position, increasing the miner's exposure to the trade pair.

## Basic Rules

1. Your miner must register on the Bittensor network to participate.
   - There is a minimal registration fee <1 TAO on mainnet.
   - There is an immunity period of 4 hours. Eliminated miners do not benefit from being in the immunity period.
2. Your miner will start in the challenge period upon entry. Miners must demonstrate consistent performance within 90 days to pass the challenge period. During this period, they will receive a small amount of TAO that will help them avoid getting deregistered. The minimum requirements to pass the challenge period:
   - Have at least 61 full days of trading
   - Don't exceed 10% max drawdown
   - Score at or above the 25th miner in each asset class in main competition. The details may be found [here](https://docs.taoshi.io/tips/p21/).
3. Miner's account size is determined from your miner's deposited collateral. Each theta deposited unlocks \$500 of trading capacity. Miners are required to deposit a minimum of 300 Theta, and are capped at a maximum of 1000 Theta deposited as collateral.
4. Miner's must select an asset category in order to submit trades. Miners will be restricted to only submitting trades in their chosen asset class.
5. Positions are uni-directional. Meaning, if a position starts LONG (the first order it receives is LONG),
   it can't flip SHORT. If you try and have it flip SHORT (using more leverage SHORT than exists LONG) it will close out
   the position. You'll then need to open a second position which is SHORT with the difference.
6. Position leverage is bound per trade pair. If an order would cause the position's leverage to exceed the upper boundary, the position leverage will be clamped. Minimum order leverage is 0.001. Crypto positional leverage limit is [0.01, 0.5]. Forex positional leverage limit is [0.1, 5].
7. Leverage is capped at 10 across all open positions in a miner's portfolio. Crypto position leverages are scaled by 10x when contributing
   to the leverage cap. <a href="https://docs.taoshi.io/tips/p10/">View for more details and examples.</a>
8. You can take profit on an open position using LONG and SHORT. Say you have an open LONG position with .5x
   leverage and you want to reduce it to a .25x leverage position to start taking profit on it. You would send in a SHORT signal
   of size .25x leverage to reduce the size of the position. LONG and SHORT signals can be thought of working in opposite
   directions in this way.
9. Miners that have passed challenge period will be eliminated for a drawdown that exceeds 10%.
10. Miners in main competition who fall below the top 25 in each asset class will be observed under a probation period. 
   - Miners in probation period have 60 days from time of demotion to be promoted back into main competition.
   - If they fail to do so within this window, they will be eliminated.
11. A miner can have a maximum of 1 open position per trade pair. No limit on the number of closed positions.
12. A miner's order will be ignored if placing a trade outside of market hours.
13. A miner's order will be ignored if they are rate limited (maliciously sending too many requests)
14. There is a 10-second cooldown period between orders of the same trade pair, during which the miner cannot place another order.
15. **CRITICAL**: Never reuse hotkeys that have been previously eliminated or deregistered. Once a hotkey is eliminated or deregistered, it is **permanently blacklisted** by the network. Validators internally track all departed hotkeys (both eliminated miners and voluntary deregistrations) and will reject orders from re-registered hotkeys. **Each registration must use a completely new, unused hotkey**. This policy ensures network integrity and prevents circumventing elimination penalties.

## Asset Class Selection
Each miner selects a single asset class to compete in (crypto or forex), and competes only against other miners with the same asset class selection. Miners who do not select an asset class are restricted from placing orders.

## Scoring Details

**Debt-Based Scoring System** (Active December 2025+)

Vanta uses a debt-based scoring system that pays miners proportionally based on their previous month's performance. The system tracks three key components for each miner:

1. **Emissions Ledger**: Records ALPHA/TAO/USD tokens earned in 12-hour checkpoints
2. **Performance Ledger**: Tracks PnL, fees, drawdown, and portfolio returns
3. **Penalty Ledger**: Applies multipliers for drawdown, risk profile, min collateral, and risk-adjusted performance

These components are combined into a **Debt Ledger** that calculates:
- **Needed Payout**: Previous month's PnL scaled by penalties (in USD)
- **Actual Payout**: Current month's emissions already received (in USD)
- **Remaining Payout**: Debt still owed to the miner (in USD)

Weights are distributed proportionally to remaining payouts, targeting completion by **day 25 of each month**. The system uses an aggressive payout strategy that front-loads emissions early in the month while respecting the hard deadline.

*Average Daily PnL* has the highest weight (90%) and incentivizes miners to maintain high returns while increasing account sizes. The remaining scoring metrics (Calmar, Sharpe, Omega, Sortino, Statistical Confidence) each contribute 2% to ensure well-rounded performance evaluation.

We calculate daily returns for all positions and the entire portfolio, spanning from 12:00 AM UTC to 12:00 AM UTC the following day. However, if a trading day is still ongoing, we still monitor real-time performance and risks. 

This daily calculation and evaluation framework closely aligns with real-world financial practices, enabling accurate, consistent, and meaningful performance measurement and comparison across strategies. This remains effective even for strategies trading different asset classes at different trading frequencies. This approach can also enhance the precision of volatility measurement for strategies.

Annualization is used for the Sharpe ratio, Sortino ratio, and risk adjusted return with either volatility or returns being annualized to better evaluate the long-term value of strategies and standardize our metrics. Volatility is the standard deviation of returns and is a key factor in the Sharpe and Sortino calculations.

In determining the correct annualization factor, we weigh more recent trading days slightly higher than older trading days. This should encourage miners to regularly update their strategies and adapt to changing market conditions, continually providing the network with the most relevant signals. The most recent 10 days account for 25% of the total score, the most recent 30 days account for 50%, and the most recent 70 days account for 75%, with a pattern that tapers exponentially over time.
The average daily PnL metric has a more aggressive recency weighting to encourage frequent trading activity. The first 10 days has 40% of the total score, the first 30 days account for 70%, and the first 70 days account for 87% also with weight that tapers exponentially over time.

Additionally, normalization with annual risk-free rate of T-bills further standardizes our metrics and allows us to measure miner performance on a more consistent basis.

### Scoring Metrics

We use **Average Daily PnL** and five risk-adjusted scoring metrics to evaluate miners based on daily returns: **Calmar Ratio**, **Sharpe Ratio**, **Omega Ratio**, **Sortino Ratio**, and **Statistical Confidence (T-Statistic)**.

The miner risk used in the risk adjusted returns is the miner’s maximum portfolio drawdown.

_Average Daily PnL_ will look at the average USD change in portfolio value for full trading days. The PnL values are based on the account sizes of miners which are calculated from deposited collateral.

_Calmar Ratio_ will look at daily returns in the prior 120 days and is normalized by the max drawdown.

$$
\text{Return / Drawdown} = \frac{(\frac{365}{n}\sum_{i=0}^n{R_i}) - R_{rf}}{\sum_i^{n}{\text{MDD}_i} / n}
$$

The _sharpe ratio_ will look at the annualized excess return, returns normalized with the risk-free rate, divided by the annualized volatility which is the standard deviation of the returns. To avoid gaming on the bottom, a minimum value of 1% is used for the volatility.

$$
\text{Sharpe} = \frac{(\frac{365}{n}\sum_{i=0}^n{R_i}) - R_{rf}}{\sqrt{\text{var}{(R) * \frac{365}{n}}}}
$$

The _omega ratio_ is a measure of the winning days versus the losing days. The numerator is the sum of the positive daily log returns while the denominator is the product of the negative daily log returns. It serves as a useful proxy for the risk to reward ratio the miner is willing to take with each day. Like the Sharpe ratio, we will use a minimum value of 1% for the denominator.

$$
\text{Omega} = \frac{\sum_{i=1}^n \max(r_i, 0)}{\lvert \sum_{i=1}^n \min(r_i, 0) \rvert}
$$

The _sortino ratio_ is similar to the Sharpe ratio except that the denominator, the annualized volatility, is calculated using only negative daily returns (i.e., losing days).

$$
\text{Sortino} = \frac{(\frac{365}{n}\sum_{i=0}^n{R_i}) - R_{rf}}{\sqrt{\frac{365}{n} \cdot \text{var}(R_i \;|\; R_i < 0)}}
$$

_Statistical Confidence_ uses a t-statistic to measure how similar the daily distribution of returns is to a normal distribution with zero mean. Low similarity means higher confidence that a miner’s strategy is statistically different from a random distribution.

$$
t = \frac{\bar{R} - \mu}{s / \sqrt{n}}
$$

| Metric                 | Scoring Weight |
|------------------------|----------------|
| Average Daily PnL      | 90%            |
| Calmar Ratio           | 2%             |
| Sharpe Ratio           | 2%             |
| Omega Ratio            | 2%             |
| Sortino Ratio          | 2%             |
| Statistical Confidence | 2%             |

### Scoring Penalties

There are two primary penalties in place for each miner:

1. Max Drawdown: Vanta eliminates miners who exceed 10% max drawdown.
2. Risk-Profiling: Miners are penalized for having positions that may create undue risk for copy traders.

To avoid the impact of a risk profiling penalty, we recommend that you avoid doing the following:

- Step three or more times into a position or increasing the max leverage twice on a losing position.
- Use more than 50% of the available leverage on the trade pair or increasing leverage by 150% relative to the entry leverage of the position
- Having uneven time intervals between orders, which would indicate they are not TWAP-scheduled orders.

Full implementation details may be found [here](https://docs.taoshi.io/tips/p19/).

The Max Drawdown penalty and Risk Profiling penalty help us detect the absolute and relative risks of a miner's trading strategy in real time.

### Fees and Transaction Costs

We want to simulate real costs of trading for our miners, to make signals from Vanta more valuable outside our platform. To do this, we have incorporated three primary costs: **Cost of Carry**, **Slippage**, and **Spread Fee**.

Cost of carry is reflective of real exchanges, and how they manage the cost of holding a position overnight. This rate changes depending on the asset class, the logic of which may be found in [our proposal 4](https://docs.taoshi.io/tips/p4/).

Slippage costs are modeled to estimate the difference between a trade's expected price (typically the last traded price or mid-price between the best bid and ask) and its actual execution price. This cost is higher for larger orders, as well as for assets with lower liquidity and higher volatility. Read more in [proposal 16](https://docs.taoshi.io/tips/p16/).

Spread fee is applied to crypto pairs only and is calculated as 0.1% multiplied by the leverage of each order. This fee simulates a transaction cost that a normal exchange would add.

##### Implementation Details

**Carry Fees:**

| Market   | Fee Period | Times                   | Rates Applied   | Triple Wednesday |
| -------- | ---------- | ----------------------- | --------------- | ---------------- |
| Forex    | 24h        | 21:00 UTC               | Mon-Fri         | ✓                |
| Crypto   | 8h         | 04:00, 12:00, 20:00 UTC | Daily (Mon-Sun) |                  |

The magnitude of the carry fees will reflect the following distribution:

| Market   | Base Rate (Annual) | Daily Rate Calculation      |
| -------- | ------------------ | --------------------------- |
| Forex    | 3%                 | 0.008% \* Max Seen Leverage |
| Crypto   | 10.95%             | 0.03% \* Max Seen Leverage  |

**Spread Fee (Transaction Fee):**

| Market   | Spread Fee Rate    | Applied To           |
| -------- | ------------------ | -------------------- |
| Forex    | None               | N/A                  |
| Crypto   | 0.1% \* Leverage   | Each order placed    |

### Leverage Limits

We also set limits on leverage usage, to ensure that the network has a level of risk protection and mitigation of naive strategies. The [positional leverage limits](https://docs.taoshi.io/tips/p5/) are as follows:

| Market   | Leverage Limit |
| -------- | -------------- |
| Forex    | 0.1x - 5x      |
| Crypto   | 0.01x - 0.5x   |

We also implement a [portfolio level leverage limit](https://docs.taoshi.io/tips/p10/), which is the sum of all the leverages from each open position. This limit is set at 10x a "typical" position, where a typical position would be 1x leverage for forex and 0.1x leverage for crypto. You can therefore open 10 forex positions at 1x leverage each, 5 forex positions at 2x leverage each, 5 forex positions at 1x and 5 crypto positions at 0.1x, etc.

## Incentive Distribution

**Debt-Based Weight Calculation** (Active December 2025+)

Starting in December 2025, Vanta uses a debt-based scoring algorithm to calculate miner weights:

1. **Previous Month Performance**: Calculate each miner's needed payout from previous month (PnL × penalties in USD)
2. **Current Month Emissions**: Sum emissions already received in current month (in USD)
3. **Remaining Debt**: Calculate remaining payout = needed - actual (in USD)
4. **Weight Assignment**: Weights are proportional to remaining debt, targeting payout completion by day 25
5. **Dynamic Dust Weights**: All miners receive minimum weights based on their challenge period status:
   - MAINCOMP: 3× dust floor (scaled up to +1 dust based on 30-day performance)
   - PROBATION: 2× dust floor (scaled up to +1 dust based on 30-day performance)
   - CHALLENGE/PLAGIARISM: 1× dust floor (scaled up to +1 dust based on 30-day performance)
   - UNKNOWN: 0× dust (no weight)
6. **Burn Address**: Excess weight (when sum < 1.0) goes to burn address (UID 229 mainnet / UID 5 testnet)

**Aggressive Payout Strategy**:
- Early month (days 1-20): Target 4-day completion to front-load emissions
- Late month (days 21-24): Target actual remaining days until deadline
- Day 25: Final payout deadline

This system ensures miners are compensated fairly based on their performance while maintaining network security through minimum weights and preventing weight concentration through the burn mechanism.

## Holidays

There are several enforced trading holidays where signals will not be processed. These include:

| Holiday       | Date         | Asset |
|---------------|--------------|-------|
| New Years     | Jan 1        | Forex |
| Good Friday   | Apr 18, 2025 | Forex |
| Christmas Day | Dec 25       | Forex |
| Boxing Day    | Dec 26       | Forex |

Where a holiday falls on a weekend, it is observed on the nearest working day.

# Easy Setup

Here are platforms that allows you to trade on Vanta with a simple interface or connect to an existing API. These facilitate trading so you can focus on building your strategy.

1. [Horizon](https://x.com/taoshiio/status/1895516351814365201)
2. [Delta Prop Shop](https://x.com/DeltaDeFi_)

# Default Setup

For our power users with more technical knowledge, we've setup some helpful infrastructure for you to send in signals to the network programatically.

The script `mining/run_receive_signals_server.py` will launch a flask server to receive order signals.
We recommend using this flask server to send in signals to the network. To see an example of sending a signal into the server, use `mining/sample_signal_request.py`.

Once a signal is properly sent into the signals server, it is parsed and stored locally in `mining/received_signals` to prepare for processing by `neurons/miner.py`. From there, the core miner logic in `neurons/miner.py` will automatically look to send the signal to validators on the network, retrying on failure. Once the signal is successfully sent into the network and ack'd by validators, the signal is stored in `mining/processed_signals`. otherwise, it gets stored in `mining/failed_signals` with debug information about which validators didn't receive the signal.

The current flow of information is as follows:

1. Run `mining/run_receive_signals_server.py` and `neurons/miner.py` to receive and parse signals
2. Send order signals from your choice of data provider (TradingView, python script, manually running `mining/sample_signal_request.py`)
3. Allow the miner to automatically send in your signals to validators
4. Validators update your existing positions, or create new positions based on your signals
5. Validators track your positions returns
6. Validators review your positions to assess drawdown every few seconds to determine if a miner should be eliminated (see main README for more info)
7. Validators wait for you to send in signals to close out positions (FLAT)
8. Validators set weights based on miner returns every 5 minutes based on portfolio performance with both open and closed positions.

When getting set up, we recommend running `mining/run_receive_signals_server.py` and `mining/sample_signal_request.py` locally to verify that order signals can be created and parsed correctly.

After that, we suggest running `mining/run_receive_signals_server.py` and `mining/sample_signal_request.py` in conjunction with `neurons/miner.py` on testnet. Inspect the log outputs to ensure that validators receive your orders. Ensure you are on your intended enviornment add the appropriate testnet flags.

| Environment | Netuid |
| ----------- | -----: |
| Mainnet     |      8 |
| Testnet     |    116 |

The simplest way to get a miner to submit orders to validators is by manually running `mining/sample_signal_request.py`. However, we expect most top miners to interface their existing trading software with `neurons/miner.py` and `mining/run_receive_signals_server.py` to automatically send trade signals. We will be releasing more detailed guides on how to set up an automated trades soon.

**DANGER**

- Do not expose your private keys.
- Only use your testnet wallet.
- Do not reuse the password of your mainnet wallet.
- Make sure your incentive mechanism is resistant to abuse.
- Your incentive mechanisms are open to anyone. They emit real TAO. Creating these mechanisms incur a lock_cost in TAO.
- Before attempting to register on mainnet, we strongly recommend that you run a miner on the testnet.
- Miners should use real exchange prices directly for training and live data purposes. This should come from MT5 and CB Pro / Binance. They should not rely on the data sources validators are providing for prices, as the data is subject to change based on potential downtime and fallback logic.

# System Requirements

- Requires **Python 3.10.**
- [Bittensor](https://github.com/opentensor/bittensor#install)

Below are the prerequisites for miners. You may be able to make a miner work off lesser specs but it is not recommended.

- 2 vCPU + 8 GB memory
- Run the miner using CPU

# Getting Started

## 1. Install Vanta

Clone repository

```bash
git clone https://github.com/taoshidev/vanta-network.git
```

Change directory

```bash
cd vanta-network
```

Create Virtual Environment

```bash
python3 -m venv venv
```

Activate a Virtual Environment

```bash
. venv/bin/activate
```

Disable pip cache

```bash
export PIP_NO_CACHE_DIR=1
```

Install dependencies

```bash
pip install -r requirements.txt
```

Note: You should disregard any warnings about updating Bittensor after this. We want to use the version specified in `requirements.txt`.

Create a local and editable installation

```bash
python3 -m pip install -e .
```

Create `mining/miner_secrets.json` and replace xxxx with your API key. The API key value is determined by you and needs to match the value in `mining/sample_signal_request.py`.

```json
{
  "api_key": "xxxx"
}
```

## 2. Create Wallets

This step creates local coldkey and hotkey pairs for your miner.

The miner will be registered to the subnet specified. This ensures that the miner can run the respective miner scripts.

Create a coldkey and hotkey for your miner wallet. A coldkey can have multiple hotkeys, so if you already have an existing coldkey, you should create a new hotkey only. Be sure to save your mnemonics!

```bash
btcli wallet new_coldkey --wallet.name <wallet>
btcli wallet new_hotkey --wallet.name <wallet> --wallet.hotkey <miner>
```

You can list the local wallets on your machine with the following.

```bash
btcli wallet list
```

## 2a. Getting Testnet TAO

### Discord

Please ask the Bittensor Discord community for testnet TAO. This will let you register your miner(s) on Testnet.

Please first join the Bittensor Discord here: https://discord.com/invite/bittensor

Then request testnet TAO here: https://discord.com/channels/799672011265015819/1331693251589312553/1331694633822060544

Bittensor Discord -> help-forum -> Requests for Testnet TAO

## 3. Register keys

This step registers your subnet miner keys to the subnet, giving it the first slot on the subnet.

```bash
btcli subnet register --wallet.name <wallet> --wallet.hotkey <miner>
```

To register your miner on the testnet add the `--subtensor.network test` and `--netuid 116` flags.

Follow the below prompts:

```bash
>> Enter netuid (0): # Enter the appropriate netuid for your environment (8 for the mainnet)
Your balance is: # Your wallet balance will be shown
The cost to register by recycle is τ0.000000001 # Current registration costs
>> Do you want to continue? [y/n] (n): # Enter y to continue
>> Enter password to unlock key: # Enter your wallet password
>> Recycle τ0.000000001 to register on subnet:8? [y/n]: # Enter y to register
📡 Checking Balance...
Balance:
  τ5.000000000 ➡ τ4.999999999
✅ Registered
```

## 4. Check that your keys have been registered

This step returns information about your registered keys.

Check that your miner has been registered:

```bash
btcli wallet overview --wallet.name <wallet>
```

To check your miner on the testnet add the `--subtensor.network test` flag

The above command will display the below:

```bash
Subnet: 8 # or 116 on testnet
COLDKEY  HOTKEY   UID  ACTIVE  STAKE(τ)     RANK    TRUST  CONSENSUS  INCENTIVE  DIVIDENDS  EMISSION(ρ)   VTRUST  VPERMIT  UPDATED  AXON  HOTKEY_SS58
wallet   miner    196    True   0.00000  0.00000  0.00000    0.00000    0.00000    0.00000            0  0.00000        *      134  none  5HRPpSSMD3TKkmgxfF7Bfu67sZRefUMNAcDofqRMb4zpU4S6
1        1        1            τ0.00000  0.00000  0.00000    0.00000    0.00000    0.00000           ρ0  0.00000
                                                                               Wallet balance: τ4.998999856
```

## 6. Run your Miner

Run the subnet miner:

```bash
python neurons/miner.py --netuid 8  --wallet.name <wallet> --wallet.hotkey <miner>
```

To run your miner on the testnet add the `--subtensor.network test` flag and override the netuuid flag to `--netuid 116`.

To enable debug logging, add the `--logging.debug` flag

You will see the below terminal output:

```bash
>> 2023-08-08 16:58:11.223 |       INFO       | Running miner for subnet: 8 on network: ws://127.0.0.1:9946 with config: ...
```

## 7. Stopping your miner

To stop your miner, press CTRL + C in the terminal where the miner is running.

# Running Multiple Miners

You may use multiple miners when testing if you pass a different port per registered miner.

You can run a second miner using the following example command:

```bash
python neurons/miner.py --netuid 116 --subtensor.network test --wallet.name <wallet> --wallet.hotkey <miner2> --logging.debug --axon.port 8095
```



# Vanta CLI: Selecting your asset class and managing collateral

Miners are required to select an asset class before submitting orders. Miners are also required to deposit a minimum of 300 Theta as collateral, up to a maximum of 1000 Theta. Asset class selection and Collateral can be managed from the Vanta CLI.

### Installing the Vanta CLI

The [Vanta CLI](https://github.com/taoshidev/vanta-cli) is included when installing Vanta. More information can be found [here](https://docs.taoshi.io/vanta/vanta-cli/). It may also be installed separately by running the following:

```bash
pip install git+https://github.com/taoshidev/vanta-cli.git
```

### Selecting an asset class with the Vanta CLI

Miners can select an asset class using the following command. Once an asset class is selected, it cannot be changed. Miners who wish to participate in multiple asset classes must register another miner.

```bash
vanta asset select
```

To select an asset class on the testnet add the `--subtensor.network test` flag

### Depositing and managing collateral

Miners can also deposit and manage collateral using the following commands:

View deposited collateral

```bash
vanta collateral list
```

Deposit collateral

```bash
vanta collateral deposit
```

Withdraw collateral. Withdrawn collateral is subject to slashing proportional the miner's current drawdown. A miner who is eliminated will have their collateral deposit slashed and burnt.

```bash
vanta collateral withdraw
```

To manage collateral on the testnet add the `--subtensor.network test` flag

# Miner Dashboard

The old local miner dashboard has been replaced by a new dashboard which can be accessed here:

- Mainnet: https://dashboard.taoshi.io
- Testnet: https://testnet.dashboard.taoshi.io

## Logging In

In order to view your miner's private positions and orders, you will need to log in to the dashboard and authenticate using a browser wallet, such as [polkadot.js](https://polkadot.js.org/extension/).

![Imgur](https://i.imgur.com/1gn58nM.png)

You may connect multiple miner hotkeys, and switch between them.

![Imgur](https://i.imgur.com/d8Yynxl.png)

Once connected, clicking the `Miner Dashboard` button will bring you to your logged in miner's page.

![Imgur](https://i.imgur.com/SrgtRpx.png)

## Important Note

The miner will only have data if validators have already picked up its orders.
A brand new miner may not have any data until after submitting an order.

# Issues?

If you are running into issues, please run with `--logging.debug` and `--logging.trace` set so you can better analyze why your miner isn't running.

# Terms of Service

We do not permit any third-party strategies to be used on the platform which are in violation of the terms and services of the original provider. Failure to comply will result in miner removal from the platform.
