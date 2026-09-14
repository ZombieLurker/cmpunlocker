#!/bin/bash
# One-shot root runbook: NVIDIA 610.57.04 + cmpunlocker v0.4 + aikitoria p2p-v3 (mixed-generation BAR1 P2P).
# Prepared 2026-09-14. Revert path: branch working-2026-09-13 + /nfs/docker/staging/cmpunlocker-backup-2026-09-13-working
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }
cd /nfs/docker/staging/cmpunlocker
[[ "$(git rev-parse --abbrev-ref HEAD)" == "v0.4-p2pv3-610.57.04" ]] || { echo "repo not on branch v0.4-p2pv3-610.57.04"; exit 1; }
echo "== 1/5 apt: move pin 610.43.* -> exactly 610.57.04, unhold, upgrade driver packages"
[[ -f /etc/apt/preferences.d/nvidia-pin.bak-610.43 ]] || cp -a /etc/apt/preferences.d/nvidia-pin /etc/apt/preferences.d/nvidia-pin.bak-610.43
sed -i "s/^Pin: version 610\\.43\\.\\*/Pin: version 610.57.04-*/" /etc/apt/preferences.d/nvidia-pin
apt-mark unhold nvidia-driver-610-open >/dev/null
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-driver-610-open nvidia-dkms-610-open nvidia-kernel-source-610-open libnvidia-compute-610 nvidia-utils-610 nvidia-firmware-610-610.57.04
apt-mark hold nvidia-driver-610-open >/dev/null
dpkg -l | grep -E "^ii\s+(nvidia-dkms-610-open|libnvidia-compute-610|nvidia-firmware-610-610)" | awk "{print \"   \", \$2, \$3}"
[[ -d /usr/src/nvidia-610.57.04 ]] || { echo "stock 610.57.04 source not present after apt"; exit 1; }
echo "== 2/5 cmpunlocker v0.4 + p2p-v3 build/install (no VFIO passthrough)"
CMPUNLOCKER_DRIVER_VERSION=610.57.04 ./install.sh --no-passthrough
echo "== 3/5 modprobe: restore Gen2 + static BAR1 (P2P) keys"
cat > /etc/modprobe.d/cmp-pcie-gen2.conf <<"MP"
options nvidia NVreg_RegistryDwords="RmForceEnableGen2=1;RMPcieLinkSpeed=0x1;RMForceStaticBar1=1"
MP
cat /etc/modprobe.d/cmp-pcie-gen2.conf
echo "== 4/5 installed module check"
KO=/lib/modules/$(uname -r)/updates/cmpunlocker/nvidia.ko
V=$(modinfo -F version $KO 2>/dev/null || true); echo "   patched nvidia.ko version: ${V:-MISSING}"
[[ "$V" == "610.57.04" ]] || { echo "ERROR: patched module is not 610.57.04 -- DO NOT REBOOT; run: sudo apt install --reinstall nvidia-dkms-610-open  (restores stock 610.57.04 modules)"; exit 1; }
ls /lib/modules/$(uname -r)/updates/dkms/ 2>/dev/null | grep -q nvidia && echo "   WARNING: stock dkms nvidia modules also present" || echo "   no stock dkms modules (good)"
[[ -f /lib/firmware/nvidia/610.57.04/gsp_ga10x.bin ]] && echo "   GSP firmware 610.57.04 present" || { echo "ERROR: 610.57.04 GSP firmware missing -- DO NOT REBOOT"; exit 1; }
modinfo -F version /lib/modules/$(uname -r)/updates/cmpunlocker/nvidia.ko 2>/dev/null || true
ls -la /usr/lib/x86_64-linux-gnu/libcuda.so.610.57.04 2>/dev/null || echo "WARNING: libcuda.so.610.57.04 not found"
echo "== 5/5 libcuda mixed-generation P2P patch (REQUIRED for 3090<->170HX: kernel patch alone leaves can_access_peer=False)"
LC=/usr/lib/x86_64-linux-gnu/libcuda.so.610.57.04
[[ -f $LC.bak ]] || cp -a "$LC" "$LC.bak"
if python3 tools/patch-libcuda-p2p.py "$LC"; then echo "   libcuda patched (backup $LC.bak)"; else echo "   libcuda: signatures not found -- already patched, or a different build; verify with the p2pmix.py test after boot"; fi
echo "== done. (old note: libcuda patch is NOT applied yet (apply after boot only if 3090<->170HX P2P is still refused):"
echo "   sudo cp /usr/lib/x86_64-linux-gnu/libcuda.so.610.57.04{,.bak} && sudo python3 tools/patch-libcuda-p2p.py /usr/lib/x86_64-linux-gnu/libcuda.so.610.57.04"
echo "NOW: cold boot  ->  sudo shutdown -h now, then power on."
