# Stable VirtualBox network for VisionBridge

The laptop's Wi-Fi address changes when it moves to a different network. The
VisionBridge cluster should therefore use a second **host-only** adapter for
communication between Windows and the three Ubuntu VMs. Keep adapter 1
(bridged) for internet access; adapter 2 remains on the laptop and has the
same addresses on every Wi-Fi network.

| Machine | Host-only address | Name used by Hadoop |
| --- | --- | --- |
| Windows host | `192.168.56.1/24` | Windows model API host |
| Primary VM (`ubuntu`) | `192.168.56.10/24` | `h-primary` |
| Worker 1 (`ubuntu-clone`) | `192.168.56.11/24` | `secondary1` |
| Worker 2 (`ubuntu-clone-2`) | `192.168.56.12/24` | `secondary2` |

## Configuration status, 25 October 2026

- Windows already has the VirtualBox host-only Ethernet adapter at
  `192.168.56.1/24`.
- VirtualBox adapter 2 is attached as **host-only** on all three VMs. Adapter
  1 remains **bridged** on each VM.
- The primary guest's `enp0s8` NetworkManager profile was set to static
  `192.168.56.10/24`. Windows reached its SSH port at that address, and the
  primary reached Qwen and BLIP on `192.168.56.1:7865/7866` with HTTP 200.
- Both worker guests now have static `enp0s8` addresses and all three guests'
  `/etc/hosts` files map `h-primary`, `secondary1`, and `secondary2` to the
  host-only addresses in the table.
- All three VMs were booted together. The primary startup monitor reported
  two live HDFS DataNodes, two active YARN NodeManagers, MongoDB, dashboard,
  and Spark refresh running. HDFS then reported zero under-replicated and zero
  missing blocks; older `/tmp` artifacts had requested replication factor 10
  and were reset to factor 2 for this two-DataNode cluster.
- The primary dashboard Docker image was rebuilt and its model URL defaults
  now point to the fixed Windows host-only address.
- VirtualBox allocations are now 4 GB RAM/8 CPUs on primary and 2 GB RAM/4
  CPUs on each worker. This fits the three VMs into the 16 GB host more
  reliably than the previous 6 GB per VM allocation. Heavy model inference
  can still compete for memory; close unrelated Windows apps for a demo.

## Recovery after an interrupted run

On 25 October, the primary VM stopped responding during a Spark refresh. An
approved reset restored SSH and the dashboard. All **19 saved MongoDB
requests** survived. Two zero-byte local JSON files were restored from those
MongoDB records; one additional zero-byte file had no MongoDB or HDFS copy and
represented an interrupted capture. The originals are preserved under
`spool/quarantine/` on the primary. No HDFS blocks were missing. The next Spark
refresh completed with 19 valid requests, 18 curated requests and 6 matched
BLIP/Qwen pairs.

The primary now has a persistent **2 GB swap file** at `/swapfile` to give its
4 GB RAM allocation more room during Spark jobs. Request images and JSON are
published atomically by `bda/capture.py` so an interrupted write cannot leave
an empty final event file. A model request that had not completed before a VM
failure may still need to be submitted again.
The Windows Qwen service is launched with `VISIONBRIDGE_OLLAMA_KEEP_ALIVE=0`,
so Ollama releases its model after an answer instead of holding it in memory
during the following Spark refresh. The next Qwen question must reload it.

## Reapply guest configuration if rebuilding a VM

The reusable script is `bda/configure_vm_hostonly.sh`. Copy it to each guest
and run it with the correct role. It configures only the second interface
`enp0s8`, keeps the bridged interface's DHCP/default route, makes a one-time
backup at `/etc/hosts.visionbridge-before-hostonly`, and writes the stable
Hadoop hostnames. It is safe to rerun.

The script is deployed on the primary at
`/home/admin/Desktop/BDA/project/bda/configure_vm_hostonly.sh`. For example:

```bash
sudo bash /home/admin/Desktop/BDA/project/bda/configure_vm_hostonly.sh primary
```

If rebuilding a worker, find its current bridged IP with `ip -br a` in that
guest, copy the script from the primary, then run it in the guest:

```bash
scp /home/admin/Desktop/BDA/project/bda/configure_vm_hostonly.sh admin@<worker-current-bridged-ip>:/tmp/visionbridge-configure-hostonly.sh
```

```bash
# Run on worker 1:
sudo bash /tmp/visionbridge-configure-hostonly.sh secondary1

# Run on worker 2:
sudo bash /tmp/visionbridge-configure-hostonly.sh secondary2
```

Do not power off a VM by force just to change its configuration. Shut it down
gracefully and wait until VirtualBox reports `poweroff`.

## Switch and verify the applications

For a fresh cluster check:

1. Verify `getent hosts h-primary secondary1 secondary2` reports `.10`,
   `.11`, `.12` on every VM.
2. Verify Windows SSH to `admin@192.168.56.10`, and primary-to-worker SSH using
   `secondary1` and `secondary2`.
3. The repository and deployed dashboard defaults for Qwen and BLIP are
   `http://192.168.56.1:7865` and `:7866`.
4. Start the three VMs, run `bash ~/Desktop/start_visionbridge_bda.sh` on the
   primary, and check two DataNodes, two NodeManagers, dashboard UP, and model
   endpoints UP.
5. Open `http://192.168.56.10:8501/` from Windows. For microphone recording,
   use the existing localhost tunnel pointed to the new stable address:

   ```powershell
   ssh -N -L 127.0.0.1:8501:127.0.0.1:8501 admin@192.168.56.10
   ```

## Rollback

Adapter 1 remains bridged, so its DHCP connection is a fallback. On any VM,
restore the backed-up hosts file with
`sudo cp /etc/hosts.visionbridge-before-hostonly /etc/hosts` if needed. To
remove the new guest profile, use `sudo nmcli connection delete
visionbridge-hostonly`. From powered-off VM settings, adapter 2 can be set
back to **None**. Do not format HDFS or replace any NameNode data while
changing network addresses.
