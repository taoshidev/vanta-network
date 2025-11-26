# Running Signals Server

This document outlines how to run the signals server to receive external signals to your miner to automatically send over the network.

## Requirements

To run the signals server, you need to first setup your API key.

Setup your `miner_secrets.json` file - inside the repository, you’ll go to the mining directory and add a file called `mining/miner_secrets.json`. Inside the file you should provide a unique API key value for your server to receive signals.

The file should look like so:

```
# replace xxxx with your API key
{
  "api_key": "xxxx"
}
```

Once you have your secrets file setup, you should keep it to reference from other systems
to send in signals.

## Running the receive signals server

First, you'll want to make sure you have your venv setup. You can do this by following the
Installation portion of the README.

Once you have your venv setup you can run the signals server. We've setup a convenient
script that will ensure your server is running at all times named `run_receive_signals_server.sh`.

You can run it with the following command inside the vanta-network directory:

`sh run_receive_signals_server.sh`

## Stopping the receive signals server

If you want to stop your server at any time
you can search for its PID and kill it with the following commands.

`pkill -f run_receive_signals_server.sh` </br>
`pkill -f run_receive_signals_server.py`

## Testing sending a signal

You can test a sample signal to ensure your server is running properly by running the
`sample_signal_request.py` script inside the `mining` directory.

1. Be sure to activate your venv
2. go to `vanta-network/mining/`
3. run `python sample_signal_request.py`
