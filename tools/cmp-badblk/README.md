# cmp-badblk

Finds bad 64 KB HBM frames on CMP 170HX cards and fences them off in the driver, so no
program is ever given that memory. Needs this branch's `driver/patches/cmp-badblk.patch`.

```bash
sudo tools/cmp-badblk/cmp-badblk.py status                  # cards, lists, config, active fences
sudo tools/cmp-badblk/cmp-badblk.py map --stop-services     # test every 170HX (500 passes), add bad frames, apply
sudo tools/cmp-badblk/cmp-badblk.py apply                   # rewrite the config for the current PCI addresses
```

After `map` or `apply` says "COLD BOOT NEEDED", power the machine off and on.

- Lists live per card UUID in `/etc/cmp-badblk/cards.json`, so a card moved to another slot
  only needs `apply` and a cold boot. `map` never removes frames: a fenced frame shows no
  errors, so it cannot be found again.
- The config is `/etc/modprobe.d/cmp-badblk.conf`
  (`NVreg_RegistryDwordsPerDevice="pci=...;RmCmpBadBlk0=0x...;..."`, frame = physical >> 16,
  at most 16 per card). The cmpunlocker `install.sh` does not touch it.
- `map` runs `badmap.cu` (built once with `nvidia/cuda:12.9.1-devel-ubuntu24.04` into
  `/var/lib/cmp-badblk/`) in a CUDA container per card. It allocates all free VRAM, runs
  write/read passes with fixed, checkerboard, address and hash patterns, logs bad words, and
  tags every 64 KB block. The script then reads the card's static BAR1 (read-only; BAR1 offset
  = FB physical address, checked against the driver's `CMP_BADBLK: ... startOffset=0x0` line)
  to find each block's physical address. Run outputs go to `/var/lib/cmp-badblk/runs/`.
- With `--stop-services` it stops llama-swap and any `vllm-*` container first and starts
  llama-swap again afterwards. Without it, it refuses to run while a 170HX has memory in use.
