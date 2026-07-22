#!/bin/bash

# Initialize variables
script="neurons/validator.py"
generate_script="runnable/generate_request_outputs.py"
# Standalone API apps (REST/WS split): own PM2 apps so an API-only deploy doesn't restart core.
rest_script="vanta_api/run_rest_server.py"
ws_script="vanta_api/run_ws_server.py"
autoRunLoc=$(readlink -f "$0")
proc_name="vanta"
generate_proc_name="generate"
rest_proc_name="vanta-rest"
ws_proc_name="vanta-ws"
args=()
generate_args=() # Assuming no specific arguments to the generate script
rest_args=()
ws_args=()
version_location="meta/meta.json"
version=".subnet_version"
start_generate=false
serve_enabled=false

old_args=$@

echo "$old_args"

# Check if pm2 is installed
if ! command -v pm2 &> /dev/null
then
    echo "pm2 could not be found. To install see: https://pm2.keymetrics.io/docs/usage/quick-start/"
    exit 1
fi

# Define your function for version comparison and other utilities here

# Checks if $1 is smaller than $2
version_less_than_or_equal() {
    [  "$1" = "`echo -e "$1\n$2" | sort -V | head -n1`" ]
}

# Checks if $1 is smaller than $2
version_less_than() {
    [ "$1" = "$2" ] && return 1 || version_less_than_or_equal $1 $2
}

get_version_difference() {
    local tag1="$1"
    local tag2="$2"

    # Extract the version numbers from the tags
    local version1=$(echo "$tag1" | sed 's/v//')
    local version2=$(echo "$tag2" | sed 's/v//')

    # Split the version numbers into an array
    IFS='.' read -ra version1_arr <<< "$version1"
    IFS='.' read -ra version2_arr <<< "$version2"

    # Calculate the numerical difference
    local diff=0
    for i in "${!version1_arr[@]}"; do
        local num1=${version1_arr[$i]}
        local num2=${version2_arr[$i]}

        # Compare the numbers and update the difference
        if (( num1 > num2 )); then
            diff=$((diff + num1 - num2))
        elif (( num1 < num2 )); then
            diff=$((diff + num2 - num1))
        fi
    done

    strip_quotes $diff
}

check_package_installed() {
    local package_name="$1"
    os_name=$(uname -s)

    if [[ "$os_name" == "Linux" ]]; then
        # Use dpkg-query to check if the package is installed
        if dpkg-query -W -f='${Status}' "$package_name" 2>/dev/null | grep -q "installed"; then
            return 1
        else
            return 0
        fi
    elif [[ "$os_name" == "Darwin" ]]; then
         if brew list --formula | grep -q "^$package_name$"; then
            return 1
        else
            return 0
        fi
    else
        echo "Unknown operating system"
        return 0
    fi
}

check_variable_value_on_github() {
    local repo="$1"
    local file_path="$2"
    local variable_name="$3"
    local branch="$4"

    local url="https://api.github.com/repos/$repo/contents/$file_path?ref=$branch"

    local response=$(timeout 30 curl -s "$url" 2>/dev/null)
    local curl_exit_code=$?

    # Check if curl timed out or failed
    if [ $curl_exit_code -ne 0 ]; then
        echo "Error: Failed to retrieve file contents from GitHub (timeout or network error)."
        return 1
    fi

    # Check if the response contains an error message
    if [[ $response =~ "message" ]]; then
        echo "Error: Failed to retrieve file contents from GitHub."
        return 1
    fi

    # Extract the base64 content and decode it
    json_content=$(echo "$response" | jq -r '.content' | base64 --decode 2>/dev/null)

    # Check if jq/base64 failed
    if [ $? -ne 0 ]; then
        echo "Error: Failed to decode GitHub response."
        return 1
    fi

    # Extract the "subnet_version" value using jq
    subnet_version=$(echo "$json_content" | jq -r '.subnet_version' 2>/dev/null)

    # Check if jq failed or returned null
    if [ $? -ne 0 ] || [ "$subnet_version" = "null" ]; then
        echo "Error: Failed to extract subnet_version from JSON."
        return 1
    fi

    # Print the value
    echo "$subnet_version"
}

strip_quotes() {
    local input="$1"

    # Remove leading and trailing quotes using parameter expansion
    local stripped="${input#\"}"
    stripped="${stripped%\"}"

    echo "$stripped"
}

read_version_value() {
    jq -r $version "$version_location"
}

requirements_file="requirements.txt"
requirements_hash_file=".requirements.txt.sha256"

compute_requirements_hash() {
    if [ ! -f "$requirements_file" ]; then
        echo ""
        return
    fi
    if command -v sha256sum &> /dev/null; then
        sha256sum "$requirements_file" | awk '{print $1}'
    else
        shasum -a 256 "$requirements_file" | awk '{print $1}'
    fi
}

pip_install_if_requirements_changed() {
    local current_hash
    current_hash=$(compute_requirements_hash)
    local stored_hash=""
    if [ -f "$requirements_hash_file" ]; then
        stored_hash=$(cat "$requirements_hash_file")
    fi

    if [ -n "$current_hash" ] && [ "$current_hash" = "$stored_hash" ]; then
        echo "requirements.txt unchanged (sha256: $current_hash). Skipping pip install -e ."
        return 0
    fi

    echo "requirements.txt changed (or no prior hash). Running pip install -e ..."
    if pip install -e .; then
        if [ -n "$current_hash" ]; then
            echo "$current_hash" > "$requirements_hash_file"
        fi
        return 0
    else
        return 1
    fi
}

check_package_installed "jq"
if [ "$?" -ne 1 ]; then
    echo "Missing 'jq'. Please install it first."
    exit 1
fi

if [ ! -d "./.git" ]; then
    echo "This installation does not seem to be a Git repository. Please install from source."
    exit 1
fi

# Loop through all command line arguments
# Similar logic to handle script arguments; adjust as necessary

while [[ $# -gt 0 ]]; do
  arg="$1"

  if [[ "$arg" == -* ]]; then
    if [[ $# -gt 1 && "$2" != -* ]]; then
      if [[ "$arg" == "--script" ]]; then
        script="$2";
        shift 2
      else
        args+=("$arg")
        args+=("$2")
        shift 2
      fi
    else
        args+=("$arg")
      shift
    fi
  else
    args+=("$arg")
    shift
  fi
done

# REST/WS split, only when --serve is set. Core gets --no-spawn-api (else its spawned copies and
# the standalone apps double-bind 48888/8765/50014/50022); --netuid/--slack-webhook-url are
# forwarded to the API apps (netuid drives is_mainnet in REST). No --serve = today's behavior.
netuid_value=""
slack_webhook_value=""
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
        --serve)
            serve_enabled=true
            ;;
        --netuid)
            netuid_value="${args[$((i + 1))]}"
            ;;
        --netuid=*)
            netuid_value="${args[$i]#*=}"
            ;;
        --slack-webhook-url)
            slack_webhook_value="${args[$((i + 1))]}"
            ;;
        --slack-webhook-url=*)
            slack_webhook_value="${args[$i]#*=}"
            ;;
    esac
done

if [ "$serve_enabled" = true ]; then
    args+=("--no-spawn-api")

    if [ -n "$netuid_value" ]; then
        rest_args+=("--netuid" "$netuid_value")
    fi
    if [ -n "$slack_webhook_value" ]; then
        rest_args+=("--slack-webhook-url" "$slack_webhook_value")
        ws_args+=("--slack-webhook-url" "$slack_webhook_value")
    fi
fi

branch=$(git branch --show-current)
echo "Watching branch: $branch"
if [ "$serve_enabled" = true ]; then
    echo "PM2 process names: $proc_name, $rest_proc_name, $ws_proc_name"
else
    echo "PM2 process names: $proc_name"
fi

current_version=$(read_version_value)

# check_and_restart_pm2 proc_name script_path args_array_name [kill_timeout_ms]
# kill_timeout_ms: PM2's default is only 1.6s before it SIGKILLs — too short for the API apps'
# graceful shutdown (close WS clients, cancel tasks, unlink shared memory), so they pass 10s.
check_and_restart_pm2() {
    local proc_name=$1
    local script_path=$2
    local -n proc_args_ref=$3
    local kill_timeout_ms=${4:-}

    # Check for current process name
    if pm2 status | grep -q $proc_name; then
        echo "The script $script_path is already running with pm2 under the name $proc_name. Stopping and restarting..."
        pm2 delete $proc_name
    fi

    # MIGRATION: Check for old "ptn" process name and stop it
    # This ensures clean migration from ptn to vanta
    if [ "$proc_name" = "vanta" ] && pm2 status | grep -q "ptn"; then
        echo "⚠️  Found old 'ptn' process from before rebrand. Stopping it..."
        pm2 delete ptn
        echo "✓ Old 'ptn' process stopped successfully"
    fi

    echo "Running $script_path with the following pm2 config:"

    # An empty args array must render [] not [''] — printf with zero args still emits one empty
    # '%s', which PM2 would pass as a literal empty-string arg that argparse rejects (crash loop).
    if [ ${#proc_args_ref[@]} -eq 0 ]; then
        joined_args=""
    else
        joined_args=$(printf "'%s'," "${proc_args_ref[@]}")
        joined_args=${joined_args%,}
    fi

    local kill_timeout_line=""
    if [ -n "$kill_timeout_ms" ]; then
        kill_timeout_line="
        kill_timeout: $kill_timeout_ms,"
    fi

    echo "module.exports = {
      apps : [{
        name   : '$proc_name',
        script : '$script_path',
        interpreter: 'python3',
        min_uptime: '5m',
        max_restarts: '5',$kill_timeout_line
        args: [$joined_args]
      }]
    }" > $proc_name.app.config.js

    cat $proc_name.app.config.js
    pm2 start $proc_name.app.config.js
}

# Start core first (owns the state-tier RPC servers), then the API apps. Ordering is a nicety,
# not a gate — the API apps lazy-connect and tolerate core being absent (readiness watchdog alerts).
pip_install_if_requirements_changed
check_and_restart_pm2 "$proc_name" "$script" args
if [ "$serve_enabled" = true ]; then
    check_and_restart_pm2 "$rest_proc_name" "$rest_script" rest_args 10000
    check_and_restart_pm2 "$ws_proc_name" "$ws_script" ws_args 10000
fi
if [ "$start_generate" = true ]; then
    check_and_restart_pm2 "$generate_proc_name" "$generate_script" generate_args
fi

backoff=1
max_backoff=60
max_retries=5

# Continuous checking and updating logic
while true; do
    # Check if current minute is divisible by 30
    current_minute=$(date +'%M')
    if [[ "$current_minute" != "07" && "$current_minute" != "37" ]]; then
        sleep 1 # Sleep for one second and check again
        continue
    fi

    retry_count=0
    latest_version=""
    current_backoff=$backoff

    echo "Starting version check at $(date)"

    while [ $retry_count -lt $max_retries ] && [ -z "$latest_version" ]; do
        retry_count=$((retry_count + 1))
        echo "Checking for latest version... (attempt $retry_count/$max_retries)"

        latest_version=$(check_variable_value_on_github "taoshidev/vanta-network" "$version_location" "$version" "$branch")

        # Check if we got a valid version (not an error message)
        if [ -n "$latest_version" ] && ! echo "$latest_version" | grep -q "^Error:"; then
            echo "Successfully retrieved latest version: $latest_version"
            break
        else
            latest_version=""  # Clear it if it was an error
            if [ $retry_count -lt $max_retries ]; then
                echo "Failed to get version. Retrying in $current_backoff seconds..."
                sleep $current_backoff
                current_backoff=$(( current_backoff * 2 ))
                if [ $current_backoff -gt $max_backoff ]; then
                    current_backoff=$max_backoff
                fi
            fi
        fi
    done

    # Check if we failed to get version after all retries
    if [ -z "$latest_version" ]; then
        echo "Failed to retrieve latest version after $max_retries attempts. Skipping this check cycle."
        sleep 300
        continue
    fi

    echo "Latest version: $latest_version"
    latest_version="${latest_version#"${latest_version%%[![:space:]]*}"}"
    current_version="${current_version#"${current_version%%[![:space:]]*}"}"

    if [ -n "$latest_version" ] && ! echo "$latest_version" | grep -q "Error" && version_less_than "$current_version" "$latest_version"; then
        echo "Updating due to version mismatch. Current: $current_version, Latest: $latest_version"
        if git pull origin "$branch"; then
            echo "New version published. Updating the local copy."
            if pip_install_if_requirements_changed; then
                echo "Package installation successful."
                # Fail-safe: a version bump restarts ALL apps. Narrowing to the changed app is a
                # future optimization — any shared-module change (vali_config, order/position
                # models, RPC serialization) needs core+rest+ws restarted together to avoid skew.
                check_and_restart_pm2 "$proc_name" "$script" args
                if [ "$serve_enabled" = true ]; then
                    check_and_restart_pm2 "$rest_proc_name" "$rest_script" rest_args 10000
                    check_and_restart_pm2 "$ws_proc_name" "$ws_script" ws_args 10000
                fi
                if [ "$start_generate" = true ]; then
                    check_and_restart_pm2 "$generate_proc_name" "$generate_script" generate_args
                fi
                current_version=$(read_version_value)
                echo "Update completed. Continuing monitoring..."
            else
                echo "Error: Package installation failed. Please check the logs."
            fi
        else
            echo "Error: Git pull failed. Please stash your changes using git stash."
        fi
    else
        echo "You are up-to-date with the latest version."
    fi

    echo "Sleeping for 300 seconds until next check..."
    sleep 300
done
