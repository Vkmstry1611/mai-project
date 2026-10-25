#!/usr/bin/env bash
# Give this VM its stable VirtualBox host-only address and shared cluster names.
# Run as: sudo bash configure_vm_hostonly.sh primary|secondary1|secondary2
set -euo pipefail

case "${1:-}" in
    primary) address=192.168.56.10 ;;
    secondary1) address=192.168.56.11 ;;
    secondary2) address=192.168.56.12 ;;
    *) echo "Usage: sudo bash $0 primary|secondary1|secondary2" >&2; exit 2 ;;
esac

if (( EUID != 0 )); then exec sudo bash "$0" "$@"; fi
interface="${VISIONBRIDGE_HOSTONLY_IFACE:-enp0s8}"
nmcli -g GENERAL.DEVICE device show "$interface" >/dev/null
connection="$(nmcli -g GENERAL.CONNECTION device show "$interface" | head -n 1)"
if [[ -z "$connection" || "$connection" == -- ]]; then
    nmcli connection add type ethernet ifname "$interface" con-name visionbridge-hostonly
    connection=visionbridge-hostonly
fi
nmcli connection modify "$connection" \
    connection.id visionbridge-hostonly connection.autoconnect yes \
    ipv4.method manual ipv4.addresses "$address/24" \
    ipv4.gateway '' ipv4.never-default yes ipv6.method disabled
nmcli connection up visionbridge-hostonly >/dev/null

if [[ ! -e /etc/hosts.visionbridge-before-hostonly ]]; then
    cp /etc/hosts /etc/hosts.visionbridge-before-hostonly
fi
python3 - <<'PY'
from pathlib import Path
import os
import tempfile

path = Path('/etc/hosts')
names = {'h-primary', 'secondary1', 'secondary2'}
kept = []
for line in path.read_text().splitlines():
    fields = line.split('#', 1)[0].split()
    if len(fields) > 1 and names.intersection(fields[1:]):
        continue
    kept.append(line)
kept += ['', '# VisionBridge stable VirtualBox host-only addresses',
         '192.168.56.10 h-primary ubuntu',
         '192.168.56.11 secondary1',
         '192.168.56.12 secondary2']
handle, temporary = tempfile.mkstemp(prefix='.hosts-visionbridge-', dir='/etc')
with os.fdopen(handle, 'w') as output:
    output.write('\n'.join(kept).rstrip() + '\n')
os.chmod(temporary, 0o644)
os.replace(temporary, path)
PY

printf 'Host-only address: %s/24 on %s\n' "$address" "$interface"
getent hosts h-primary secondary1 secondary2
