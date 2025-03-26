#!/bin/bash

MAIN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

function nice_echo {
        case $1 in
                strong)
                        printf "\033[1;36m%s\033[0m\n" "$2" # Light Cyan
                        ;;
                docker)
                        printf "\033[1;34m%s\033[0m\n" "$2" # Light Blue
                        ;;
                error)
                        printf "\033[0;33m%s\033[0m\n" "$2" # Brown/Orange
                        ;;
                normal|*)
                        printf "\033[1;35m%s\033[0m\n" "$2" # Light Purple
                        ;;
        esac
}

function error_stop {
        nice_echo "error" " <!> Error, stopping scripts <!>"
        exit 1
}

function install {
        if ! [ -x "$(command -v pyenv)" ]; then
                brew install pyenv
                echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
                echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
                echo 'eval "$(pyenv init -)"' >> ~/.zshrc
                source ~/.zshrc
        fi
        pyenv install --skip-existing
        if ! [ -x "$(command -v poetry)" ]; then
                curl -sSL https://install.python-poetry.org | python3 -
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
        fi
        poetry self update # Use the latest version of poetry
        nice_echo "normal" "-> Installing dependencies"
        poetry update --lock # In order to get the latest versions of the dependencies and to update the poetry.lock file
        poetry sync --no-root --all-extras
}

function linter {
        nice_echo "strong" "-> Linting code"
        poetry run ruff check . --fix
        poetry run black . 
        nice_echo "normal" "-> running MyPy type checker"
        poetry run mypy --no-namespace-packages --disallow-untyped-decorators .
}

function usage {
        echo "Usage: $0 <COMMAND> [PARAMS]"
        echo ""
        echo "Commands :"
        echo " - ACTION values :"
        echo "   * install                                              - Install environment."
        echo "   * linter                                               - Run linter, formatter and code checker (ruff, black, and mypy)."
}

# Checking parameters and Env
if [[ "$1" == "" ]]; then
   echo "Missing arguments."
   usage
   exit 1
fi
case "$1" in
        install)
                install
                ;;
        linter)
                linter
                ;;
        *)
                echo "Invalid COMMAND detected (${1})"
                usage
                exit 1
                ;;
esac
