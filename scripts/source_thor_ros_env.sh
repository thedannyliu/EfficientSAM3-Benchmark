#!/usr/bin/env bash

# Source this file, do not execute it:
#   source scripts/source_thor_ros_env.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "source this script instead of executing it: source scripts/source_thor_ros_env.sh" >&2
  exit 1
fi

_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_repo_root="$(cd -- "${_script_dir}/.." && pwd)"

export THOR_ROS_SETUP="${THOR_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
export THOR_VENV="${THOR_VENV:-${HOME}/venvs/effisam3_venv_ros}"
export SAM3_SOURCE="${SAM3_SOURCE:-${HOME}/efficientsam3/sam3}"

source "${THOR_ROS_SETUP}"
source "${THOR_VENV}/bin/activate"

_venv_site="$("${THOR_VENV}/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

_prepend_pythonpath() {
  if [[ -z "${1:-}" ]]; then
    return
  fi
  case ":${PYTHONPATH:-}:" in
    *":${1}:"*) ;;
    *) export PYTHONPATH="${1}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac
}

_prepend_pythonpath "${_venv_site}"
if [[ -d "${SAM3_SOURCE}" ]]; then
  _prepend_pythonpath "${SAM3_SOURCE}"
else
  echo "warning: SAM3_SOURCE does not exist: ${SAM3_SOURCE}" >&2
fi
_prepend_pythonpath "${_repo_root}"

if [[ -f "${_repo_root}/ros_ws/install/setup.bash" ]]; then
  source "${_repo_root}/ros_ws/install/setup.bash"
fi

unset -f _prepend_pythonpath
unset _script_dir _repo_root _venv_site
