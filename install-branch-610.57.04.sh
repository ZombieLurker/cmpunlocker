#!/bin/bash
# Install the currently checked-out cmpunlocker branch on the running 610.57.04 driver (no apt step).
# Prepared 2026-10-04 for the 74 SM / ECC work. Revert: /nfs/docker/staging/cmpunlocker-backup-2026-10-04-70sm/README.md
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }
cd /nfs/docker/staging/cmpunlocker
BR=$(git -c safe.directory="*" rev-parse --abbrev-ref HEAD); echo "== branch $BR ($(git -c safe.directory="*" rev-parse --short HEAD))"
[[ "$BR" == v0.4-p2pv3-74sm-610.57.04 || "$BR" == v0.4-p2pv3-74sm-ecc-610.57.04 || "$BR" == v0.4-p2pv3-74sm-badblk-610.57.04 ]] || { echo "unexpected branch"; exit 1; }
LOG=logs/install-$BR-$(date +%Y%m%dT%H%M%S).log; mkdir -p logs
echo "== 1/4 build + install (no VFIO passthrough), log $LOG"
CMPUNLOCKER_DRIVER_VERSION=610.57.04 ./install.sh --no-passthrough 2>&1 | tee "$LOG"
echo "== 2/4 modprobe: Gen2 + static BAR1 (P2P) keys; depmod override"
cat > /etc/modprobe.d/cmp-pcie-gen2.conf <<"MP"
options nvidia NVreg_RegistryDwords="RmForceEnableGen2=1;RMPcieLinkSpeed=0x1;RMForceStaticBar1=1"
MP
cmp -s /etc/depmod.d/cmpunlocker.conf /nfs/docker/staging/cmpunlocker-backup-2026-10-04-70sm/etc/cmpunlocker.conf \
  || { cp /nfs/docker/staging/cmpunlocker-backup-2026-10-04-70sm/etc/cmpunlocker.conf /etc/depmod.d/cmpunlocker.conf; echo "   restored depmod override"; }
depmod -a "$(uname -r)"
echo "== 3/4 module checks"
D=/lib/modules/$(uname -r)/updates/cmpunlocker
V=$(modinfo -F version $D/nvidia.ko); echo "   nvidia.ko version $V"; [[ "$V" == 610.57.04 ]] || { echo "ERROR: wrong module version, DO NOT REBOOT"; exit 1; }
[[ "$(modinfo -n nvidia)" == "$D/nvidia.ko" ]] && echo "   modprobe resolves nvidia to $D (good)" || { echo "ERROR: modprobe resolves nvidia to $(modinfo -n nvidia), DO NOT REBOOT"; exit 1; }
grep -aq "SM-RECONFIG" $D/nvidia.ko && echo "   74 SM code present" || { echo "ERROR: 74 SM code missing"; exit 1; }
[[ "$BR" == *ecc* ]] && echo "   ECC branch installed" || true
grep -aq "CMP_BADBLK" $D/nvidia.ko && echo "   bad-block fencing code present" || true
echo "== 4/4 libcuda mixed-generation P2P patch"
LC=/usr/lib/x86_64-linux-gnu/libcuda.so.610.57.04
cmp -s $LC /nfs/docker/staging/cmpunlocker-backup-2026-10-04-70sm/libcuda.so.610.57.04.patched && echo "   libcuda still the patched copy (good)" \
  || { python3 tools/patch-libcuda-p2p.py "$LC" || echo "   libcuda patch tool: check output"; }
echo "== done. Reboot next."
