#!/usr/bin/env bash
# Load the user-owned API configuration without storing secrets in the repository.

vision_avoid_api_env_file() {
    if [[ -n "${VISION_AVOID_API_ENV:-}" ]]; then
        printf '%s\n' "$VISION_AVOID_API_ENV"
    else
        printf '%s\n' "${XDG_CONFIG_HOME:-$HOME/.config}/vision_avoid/api.env"
    fi
}

vision_avoid_load_api_env() {
    local env_file
    env_file="$(vision_avoid_api_env_file)"
    if [[ ! -r "$env_file" ]]; then
        printf 'API config is missing or unreadable: %s\n' "$env_file" >&2
        printf 'Create it with mode 600; see README.md section 可选远端 API.\n' >&2
        return 2
    fi

    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a

    if [[ -n "${VLM_API_KEY_FILE:-}" ]]; then
        if [[ ! -r "$VLM_API_KEY_FILE" ]]; then
            printf 'API key file is missing or unreadable: %s\n' "$VLM_API_KEY_FILE" >&2
            return 2
        fi
        local key_env="${VLM_API_KEY_ENV:-DASHSCOPE_API_KEY}"
        local key_value
        key_value="$(<"$VLM_API_KEY_FILE")"
        if [[ -z "$key_value" ]]; then
            printf 'API key file is empty: %s\n' "$VLM_API_KEY_FILE" >&2
            return 2
        fi
        printf -v "$key_env" '%s' "$key_value"
        export "$key_env"
    fi

    export VLM_PROVIDER="${VLM_PROVIDER:-api}"
    export VLM_API_ENABLED="${VLM_API_ENABLED:-true}"
}
