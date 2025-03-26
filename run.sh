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
                brew install poetry
        fi
        poetry self update # Use the latest version of poetry
        if ! [ -x "$(command -v az)" ]; then
                brew install azure-cli
                az login
        fi
        nice_echo "normal" "-> Installing dependencies"
        sub_install
        poetry update --lock # In order to get the latest versions of the dependencies and to update the poetry.lock file
        poetry sync --no-root --all-extras
        if [ ! -e ".env" ]; then
            cp .env.dist .env
        fi
        nice_echo "normal" "-> Launching Docker-Compose environment for Opensearch, kafka..."
        docker compose -f docker-compose.yml up -d || error_stop
        nice_echo "normal" "-> Running database migrations"
        poetry run alembic upgrade head
        sleep 10  # Here because sometimes the container for the Open search in local takes time to spin up, so the data loader script (below) would fail
        nice_echo "normal" " -> Starting data loader script"
        poetry run python automation-copilots/automation_copilots_tests/scripts/os_data_loader.py
        poetry run python automation-copilots/automation_copilots_tests/scripts/pg_init.py
}

function sub_install {
        directories=("automation-toolkit" 
                     "automation-copilots"
                     "smith"
                     "projects/argilla-project"
                     "projects/studies/prompt-compression"
                     "projects/os-gradio-ui"
                     )
        for dir in "${directories[@]}"; do
                nice_echo "normal" "-> Installing Sub dependencies for $dir"
                cd ./$dir
                poetry update --lock
                cd $MAIN_DIR
        done
}

function start() {
        to_run="unknown"
        docker_compose="true"
        case $1 in
            "copilot_api")
                    to_run="./automation-copilots/automation_copilots/application/copilot_api.py"
                    ;;
            "knowledge_worker")
                    to_run="./automation-copilots/automation_copilots/application/knowledge_worker.py"
                    ;;
        esac

        if [ "$to_run" = "unknown" ]; then
          nice_echo "error" "Missing app argument. Use one of the following options: copilot_api, knowledge_worker"
          exit 0
        fi

        nice_echo "strong" "-> Starting local environment"
        nice_echo "normal" "-> Exporting environment variables"
        # shellcheck disable=SC2046
        export $(grep -v '^#' .env | xargs)

        if [ "$docker_compose" = "true"  ]; then
           nice_echo "normal" "-> Launching Docker-Compose environment for Opensearch, kafka..."
           docker compose -f docker-compose.yml up -d || error_stop
        fi

        nice_echo "normal" "-> Starting the server"
        poetry run python $to_run
}

function tests {
        to_run=""
        docker_compose="true"
        case $1 in
            "copilot_api")
                    to_run="./automation-copilots/automation_copilots_tests/copilot"
                    ;;
            "knowledge_worker")
                    to_run="./automation-copilots/automation_copilots_tests/knowledge_worker"
                    ;;
        esac
        nice_echo "normal" "-> Exporting environment variables"
        # shellcheck disable=SC2046
        export $(grep -v '^#' .env | xargs)
        nice_echo "normal" "-> Launching Docker-Compose environment for Opensearch, kafka... for tests"
        docker compose -f docker-compose.yml up -d || error_stop
        nice_echo "strong" "-> Running tests"
        poetry run pytest $to_run -vv --diff-symbols --ignore=./automation-copilots/automation_copilots_tests/no_reg --ignore=./smith  --ignore=./projects/studies --durations=20 -n=5 --dist=loadfile
}

function test {
        to_run=""
        docker_compose="true"
        nice_echo "strong" "-> Running tests" + $1
        poetry run pytest -vv $1
}

function no_reg_tests {
        nice_echo "strong" "-> Running no-reg tests"
        environment=".env"
        file=""
        method=""
        for arg in "$@"; do
            case $arg in
                --file_name=*)
                    file="${1#*=}"
                    ;;
                --method=*)
                    method="${2#*=}"
                    ;;
                *)
            esac
        done

        set -a
        source $environment
        set +a

        if  [ ! -z $file ] && [ ! -z $method ] # file and method specified
        then
        :
            poetry run pytest ./automation-copilots/automation_copilots_tests/no_reg/$file -k $method
        elif [ ! -z $file ] # file specified
        then
        :
            poetry run pytest ./automation-copilots/automation_copilots_tests/no_reg/$file
        else
        :
            # -n=3 is the number of workers, from experience more than 3 would cause the code to hit some rate limits and would slow everything down
            poetry run pytest ./automation-copilots/automation_copilots_tests/no_reg --durations=10 -n=3
        fi

}

function stop {
        nice_echo "strong" "-> Stopping local environment"
        docker compose -f docker-compose.yml stop
}

function kill {
        nice_echo "strong" "-> Killing running app.py"
        ps aux | grep copilot_api.py | grep -v "grep" | awk '{ print $2 }' | xargs -I % kill -9 %
        ps aux | grep knowledge_worker.py | grep -v "grep" | awk '{ print $2 }' | xargs -I % kill -9 %
}

function linter {
        nice_echo "strong" "-> Linting code"
        directories=("./projects/studies" \
                     "./migrations" \
                     "./automation-toolkit" \
                     "./automation-copilots" \
                     "./smith"
                     )
        for dir in "${directories[@]}"; do
                FILE=$dir"/pyproject.toml"
                nice_echo "normal" "-> running ruff and black for $dir"
                if [ -e "$FILE" ]; then
                    cd $dir
                    poetry run ruff check . --fix
                    poetry run black . 
                    cd $MAIN_DIR
                else
                    poetry run ruff check $dir --fix
                    poetry run black $dir
                fi         
        done
        nice_echo "strong" "-> running vulture"
        poetry run vulture ./
        directories=("./projects/studies/speed_test" \
                     "./migrations" \
                     "./projects/studies/prompt-compression/prompt_compression" \
                     "./automation-toolkit" \
                     "./automation-copilots" \
                     "./smith"
                     )
        for dir in "${directories[@]}"; do
                FILE=$dir"/pyproject.toml"
                nice_echo "normal" "-> running MyPy type checker for $dir"
                if [ -e "$FILE" ]; then
                    cd $dir
                    poetry run mypy --no-namespace-packages --disallow-untyped-decorators .
                    cd $MAIN_DIR
                else
                    poetry run mypy --no-namespace-packages --disallow-untyped-decorators $dir
                fi
        done
}

function get_embedding {
        INPUT_STRING="$1"
        nice_echo "strong" "-> Get embedding for string : $INPUT_STRING"
        nice_echo "normal" "-> Configuring environment"
        nice_echo "normal" "-> Activating venv"
        source venv/bin/activate
        nice_echo "normal" "-> Exporting environment variables"
        export $(grep -v '^#' .env | xargs)
        export PYTHONPATH=$PYTHONPATH:./automation-copilots/automation_copilots
        python automation-copilots/automation_copilots/scripts/get_embedding.py "$INPUT_STRING"
}

function schema_migration {
        migration_to_run=$1
        nice_echo "normal" "-> Exporting environment variables"
        # shellcheck disable=SC2046
        export $(grep -v '^#' .env | xargs)
        poetry run python automation-copilots/automation_copilots/scripts/update_opensearch_mappings.py $1
}

function usage {
        echo "Usage: $0 <COMMAND> [PARAMS]"
        echo ""
        echo "Commands :"
        echo " - ACTION values :"
        echo "   * install                                              - Install environment."
        echo "   * start                                                - Launch environment and start service."
        echo "   * stop                                                 - Stop environment."
        echo "   * tests <opt:service_name>                             - Run tests for a specific service. Run all tests otherwise."
        echo "   * test <test_file_path>                                - Runs a single test; give the full path from the root of the project"
        echo "   * no_reg_tests --file_name --method                    - Run no regression tests"
        echo "   * linter                                               - Run linter, formatter and code checker (ruff, black, and mypy)."
        echo "   * kill                                                 - Kill running app.py"
        echo "   * get_embedding <input_string>                         - Get embedding from input_string"
        echo "   * schema_migration <migration>                         - Run the schema migration script"
}

function database_migration {
        nice_echo "normal" "-> Running database migrations"
        poetry run alembic upgrade head
}

function extract_logs {
        nice_echo "normal" "-> Extracting logs..."
        source .env
        python automation-copilots/automation_copilots/scripts/datadog/extract_logs_from_datadog_sdk.py $1
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
        start)
                start $2
                ;;
        stop)
                stop
                ;;
        kill)
                kill
                ;;
        linter)
                linter
                ;;
        test)
                test $2
                ;;
        tests)
                tests $2
                ;;
        no_reg_tests)
                no_reg_tests $2 $3
                ;;
        get_embedding)
                get_embedding "$2"
                ;;
        schema_migration)
                schema_migration "$2"
                ;;
        database_migration)
                database_migration
                ;;
        extract_logs)
                extract_logs $2
                ;;
        *)
                echo "Invalid COMMAND detected (${1})"
                usage
                exit 1
                ;;
esac
