> Proprietary Trading Network is now Vanta Network!

<p align="center">
  <a href="https://www.vantanetwork.io">
    <img width="385" alt="Vanta Network logo" src="https://www.taoshi.io/white-black.png">
  </a>
</p>

<div align='center'>

[![Discord Chat](https://img.shields.io/discord/1163496128499683389.svg)](https://discord.gg/vantatrading)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

</div>

<p align="center">
  <a href="https://www.vantanetwork.io">Vanta Network</a>
  ·
  <a href="https://www.vantatrading.io">Vanta Trading</a>
  ·
  <a href="#get-started">Installation</a>
  ·  
  <a href="https://www.vantanetwork.io/dashboard">Dashboard</a>
  ·
  <a href="https://x.com/VantaNetworkSN8">Twitter</a>
    ·
  <a href="https://www.bittensor.com">Bittensor</a>
</p>

---

<details>
  <summary>Table of contents</summary>
  <ol>
    <li><a href="#vanta-network">Vanta Network</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#how-does-it-work">How does it work?</a></li>
    <li>
      <a href="#get-started">Get Started</a>
    </li>
    <li><a href="#building-a-strategy">Building a Strategy</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>

  </ol>
</details>

---

<details id='bittensor'>
  <summary>What is Bittensor?</summary>

Bittensor is a mining network, similar to Bitcoin, that includes built-in incentives designed to encourage computers to provide access to machine learning models in an efficient and censorship-resistant manner. Bittensor is comprised of Subnets, Miners, and Validators.

> Explain Like I'm Five

Bittensor is an API that connects machine learning models and incentivizes correctness through the power of the blockchain.

### Subnets

Subnets are decentralized networks of machines that collaborate to train and serve machine learning models.

### Miners

Miners run machine learning models. They send signals to the Validators.

### Validators

Validators recieve trade signals from Miners. Validators ensure trades are valid, store them, and track portfolio returns. 

</details>

<br />
<br />

# Vanta Network

This repository contains the code for the Vanta Network developed by Taoshi.

Vanta receives signals from quant and deep learning machine learning trading systems to deliver the world's
most complete trading signals across a variety of asset classes.

# Features

🛠️&nbsp;Open Source Strategy Building Techniques (In Our Taoshi Community)<br>
🫰&nbsp;Signals From a <a href="https://github.com/taoshidev/vanta-network/blob/main/vali_objects/trade_pair.py#L46"> Variety of Asset Classes</a> - Forex, Crypto, Equities, and Commodities<br>
📈&nbsp;<a href="https://tokenomics.taoshi.io">Millions of $ Funding</a> to Top Traders<br>
💪&nbsp;Innovative Trader Performance Metrics that Identify the Best Traders<br>
🔎&nbsp;<a href="https://www.vantanetwork.io/dashboard">Trading + Metrics Visualization Dashboard</a><br>
🔎&nbsp;Maximum <a href="https://www.vantanetwork.io/transparency">Transparency</a> for all updates

## How does it work?

Vanta is the most challenging & competitive network in the world. Our miners need to provide futures based signals (long/short)
that are highly efficient and effective across various markets to compete (forex, crypto, equities, commodities). The top miners are
those that provide the most returns, while never exceeding certain drawdown limits.

### Rules

1. Miners can submit LONG, SHORT, or FLAT signal for Forex, Crypto, Equities, or Commodities trade pairs into the network during market hours. <a href="https://github.com/taoshidev/vanta-network/blob/main/vali_objects/trade_pair.py#L125">Currently supported trade pairs</a>
2. Miners are eliminated if they are detected as plagiarising other miners, if they exceed a 5% intraday drawdown from the day's opening equity, an 8% end-of-day drawdown from their highest-ever end-of-day equity (high-water mark), or if they go 60 days without submitting a single order (more info in the "Eliminations" section).
3. There is a fee for leaving positions open "carry fee". The fee is equal to 10.95%/3% per year for a 1x leverage position (crypto/forex respectively); equities instead pay a 3%/yr stock-borrow fee (short) or 6.6%/yr margin interest on the borrowed amount (long). Positions in Hyperliquid-sourced trade pairs (most crypto pairs, commodities, indices, and some equities) pay live Hyperliquid funding rates instead of the flat rates above <a href="https://docs.taoshi.io/tips/p4/">More info</a>
4. There is a spread (transaction) fee applied to crypto, equities, commodities, and indices orders, calculated as a percentage of order value - 0.05% for crypto and equities, 0.045% for commodities and indices (forex has no spread fee). This simulates a transaction cost that a normal exchange would add.
5. There is a slippage assessed per order. The slippage cost is is greater for orders with higher leverages, and in assets with lower liquidity.
6. Miners are rewarded using a debt-based scoring system that tracks their emissions, performance, and penalties. Weights are set based on the previous week's performance (PnL scaled by penalties), with payout periods starting and ending at midnight UTC on Sunday <a href="https://github.com/taoshidev/vanta-network/blob/main/docs/miner.md">More info</a>

With this system only the world's best traders & deep learning / quant based trading systems can compete.


# Eliminations

In the Vanta Network, eliminations occur for miners that commit plagiarism, breach drawdown limits, fail to exit probation in time, or are inactive.


### Plagiarism Eliminations

Miners who repeatedly copy another miner's trades will be eliminated. Our system analyzes the uniqueness of each submitted order. If an order is found to be a copy (plagiarized), it triggers the miner's elimination.

### Max Drawdown Elimination

Miners who exceed a 5% intraday drawdown (measured from the day's opening equity) or an 8% end-of-day drawdown (measured from their highest-ever end-of-day equity) will be eliminated. Our system continuously tracks each miner's equity to enforce these limits and maintain risk control.

Entity subaccounts (see <a href="https://github.com/taoshidev/vanta-network/blob/main/docs/entity_miner.md">Entity Miner docs</a>) follow a related but distinct drawdown rule set, assigned per-subaccount at creation: either the same trailing intraday/EOD rules as regular miners, or a static rule that eliminates if balance or 00:00 UTC equity drops more than 5% below the subaccount's starting balance.

### Probation Elimination

Miners who rank below the 25th highest ranking miner in each asset class will be observed in a probationary period. From that point, they have 90 days to achieve a rank of 25 or better in their asset class. If they fail to do so within that window, they will be eliminated.

### Inactivity Elimination

Miners in challenge period, main competition, or probation who go 60 days without submitting a single order will be eliminated. This also applies to entity-miner subaccounts, though not to the entity hotkey itself, which never submits orders directly.

### Post-Elimination

After elimination, miners are not immediately deregistered from the network. They will undergo a waiting period, determined by registration timelines and the network's immunity policy, before official deregistration. Upon official deregistration, the miner forfeits registration fees paid.

### Hotkey Blacklisting

**IMPORTANT**: Once a hotkey is eliminated or deregistered from the network, it is **permanently blacklisted** and cannot be re-registered. The network internally tracks all departed hotkeys (both eliminated and voluntarily deregistered) in a frozen/blacklisted state.

If you attempt to re-register a previously used hotkey after elimination or deregistration:
- Your orders will be rejected by validators
- You will not be able to participate in the network
- You will need to create and register a **completely new hotkey** to participate again

**Each registration requires a fresh, unused hotkey.** This policy ensures network integrity and prevents circumventing elimination penalties.



# Get Started

### Mainnet Trade Dashboard
Take a look at the top traders on Vanta <a href="https://www.vantanetwork.io/dashboard">Dashboard</a>

### Auto Trade with Vanta data 
https://x.com/glitchfinancial

### Theta Token
https://www.taoshi.io/theta

### Validator Installation

Please see our [Validator Installation](https://github.com/taoshidev/vanta-network/blob/main/docs/validator.md) guide.

### Miner Installation

Please see our [Miner Installation](https://github.com/taoshidev/vanta-network/blob/main/docs/miner.md) guide.

# Building a strategy

We recommend joining our community hub via Discord to get assistance in building a trading strategy. Analysis and information
on how to build a deep learning ML based strategy will continue to be discussed in an open manner by team Taoshi to help
guide miners to compete.

# Contributing

To contribute to Taoshi, open a pull request or file an issue on this repository. For support, join us in the [Vanta Trading Discord](https://discord.gg/vantatrading).

# License

Bittensor's source code in this repository is licensed under the MIT License.
Taoshi Inc's source code in this repository is licensed under the MIT License.
