# Send Signals from TradingView to Taoshi's Vanta Network

## Overview

This document provides traders the ability to send signals directly from TradingView to the Vanta Network to compete as a miner.

Please start by reviewing our <a href="https://github.com/taoshidev/proprietary-trading-network">README</a> to better understand Vanta.

If you have questions, you can join our <a href="https://discord.gg/MWWqaH3VJU">Taoshi Community Hub on Discord</a>
for us to help guide you.

## Requirements

1. You’ll need a cloud server with open ports necessary to receive signals (port 80 by default).
2. You’ll need to define a specific API key for access
3. You’ll need the Premium version of TradingView

## Steps

We’ll start by first having you test sending signals using testnet. We’ll start by first setting up your testnet miner.

### Setting Up Testnet Miner

Once you’ve reviewed our README, we recommend setting up your miner on testnet. You can do so by
following our <a href = "https://github.com/taoshidev/proprietary-trading-network/blob/main/docs/miner.md">installation guide</a> with the testnet flag.

Once you have a testnet miner registered via a wallet setup, you can test out sending your signals to your
testnet miner using the following steps.

1. **Setup your miner’s server** - You should setup your miner’s server in a cloud service provider. You should ensure
   your server is capable of receiving requests via open ports. By default, the expected port will be 80 to receive
   signals for your server.

2. **Follow documentation on running the receive signals server** - We’ve created <a href="https://github.com/taoshidev/proprietary-trading-network/blob/main/docs/running_signals_server.md">documentation</a> on how to run the receive
   signals server for you. Follow the guide to run your server.

3. **Ensure your server can receive signals** - We recommend cloning the proprietary-trading-network repository onto
   another machine (locally) & running the `sample_signal_request.py` script inside of it pointing to your server’s endpoint. First, make
   sure you update the API key in the repo to be what the server is expecting (updating `mining/miner_secrets.json`). Next you can run
   the `sample_signal_request.py` script passing in an argument for your server's endpoint `python sample_signal_request.py http://example.com:80`. This will ensure your
   setup is correct, and your signals server is capable and ready to receive external signals.

4. **Run your miner on testnet** - If you didn’t already, ensure your miner is running on testnet. You can do so by
   following the README’s guide on how to run your miner.

With the four steps above, you should have your signals server ready to receive signals

## Setting Up TradingView Signals

Now that your miner is ready to receive signals, we can send TradingView signals to your server using your
server’s endpoint and your designated API key.

**NOTE -** Your TradingView signals should pass a leverage value. You can do this using `comment` as part of orders in Pine scripts.
This will allow you to specify the leverage you used on a per order basis. This is something you'll need to add to every
order that's inside of your pine script.

1. Inside of TradingView go to your Alerts page using the right hand bar (the image of a clock).
2. Add your strategy to the chart for the trade pair you want to generate signals on.
3. Choose "Create Alert" using the + in the top right corner of the alerts panel.
4. In the Alert's Settings tab, choose the strategy you added to the chart using the "Condition" drop down. Set an appropriate name for the alert.
5. In the Alert's Settings tab add the following as your message where API Key is your API Key you've chosen, and the trade pair is the trade pair for the strategy you want to send signals on.

```
{"api_key":"xxxx","order_type":"{{strategy.market_position}}","trade_pair":"BTCUSD","leverage":"{{strategy.order.comment}}"}
```

6. In the Alert's Notifications tab, go to Webhook URL and put in your server's address where it will receive signals.
   This should be the same address you used for testing mentioned above in Step 3 of Setting up testnet miner.
7. Once complete, your Alert should show up as "Active" in the Alerts panel on the right hand side of your TradingView
   account's page.

Your TradingView account should now be ready to send in signals to your server. Below are some images to help guide you
in setting up your alert.

<img width="600" alt="Screenshot 2024-03-19 at 6 02 24 PM" src="https://github.com/taoshidev/proprietary-trading-network/assets/68529441/1b6a0198-a875-4881-a7d7-2056d3e02ac2">
<img width="600" alt="Screenshot 2024-03-19 at 1 26 40 PM" src="https://github.com/taoshidev/proprietary-trading-network/assets/68529441/22cfa76f-7a8f-4db9-a9eb-ca5630c39f61">


Note:
You will want to silo access to your new Vanta signals server on port 80 so that only TradingView can access it.

See here for the latest <a href="https://www.tradingview.com/support/solutions/43000529348-about-webhooks/">IPv4 TradingView addresses</a>.

At the time of writing, they can be whitelisted like this for example:

```bash
sudo ufw allow from 52.89.214.238 to any port 80;
sudo ufw allow from 34.212.75.30 to any port 80;
sudo ufw allow from 54.218.53.128 to any port 80;
sudo ufw allow from 52.32.178.7 to any port 80;
```
