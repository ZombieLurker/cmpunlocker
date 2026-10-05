#!/usr/bin/env python3
"""cmp-badblk: find bad 64 KB HBM frames on CMP 170HX cards and fence them off in the driver.

Needs the cmpunlocker build with cmp-badblk.patch (it reads the per-GPU registry keys
RmCmpBadBlk0..15 at boot and blacklists those frames in the allocator).

  sudo cmp-badblk.py status                 cards, stored lists, config, boot-log state
  sudo cmp-badblk.py map [--passes N] [--stop-services]
                                            test every 170HX, add any bad frames to the list,
                                            then run 'apply'
  sudo cmp-badblk.py apply                  write /etc/modprobe.d/cmp-badblk.conf for the
                                            cards' current PCI addresses

Lists are stored per card UUID in /etc/cmp-badblk/cards.json, so moving a card to another
slot only needs 'apply' and a cold boot. 'map' only ever adds frames: a card that is already
fenced shows no errors on its fenced frames, so existing entries are kept.

How 'map' works: badmap.cu (run in a CUDA container on one card) allocates all free VRAM,
runs write/read-back passes, logs bad words and tags every 64 KB block. This script then
reads the card's BAR1 (static mapping, BAR1 offset = FB physical address) read-only to find
where each tagged block sits physically.
"""
import argparse, collections, csv, ctypes, datetime, json, os, re, shutil, struct, subprocess, sys, time

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = "/var/lib/cmp-badblk"
DB_PATH = "/etc/cmp-badblk/cards.json"
CONF_PATH = "/etc/modprobe.d/cmp-badblk.conf"
IMG_DEVEL = "nvidia/cuda:12.9.1-devel-ubuntu24.04"
IMG_RUNTIME = "nvidia/cuda:12.9.1-runtime-ubuntu24.04"
LLAMA_SWAP_DIR = "/nfs/docker/llama-swap"
MAX_KEYS = 16                      # RmCmpBadBlk0..15 in cmp-badblk.patch
MAGIC0, MAGIC1 = 0xB4D3A9C1, 0x5EEDF00D
BLOCK = 64 * 1024


def run(cmd, check=True, **kw):
    return subprocess.run(cmd, check=check, text=True, capture_output=True, **kw)


def cards():
    """CMP 170HX cards as dicts: index, pci (DDDD:BB:DD.F), uuid, serial, used_mib."""
    out = run(["nvidia-smi", "--query-gpu=index,name,pci.bus_id,uuid,serial,memory.used",
               "--format=csv,noheader,nounits"]).stdout
    res = []
    for line in out.strip().splitlines():
        idx, name, bus, uuid, serial, used = [x.strip() for x in line.split(",")]
        if "CMP 170HX" in name:
            res.append(dict(index=int(idx), pci=bus[-12:].lower(), uuid=uuid, serial=serial, used_mib=int(used)))
    return res


def load_db():
    if os.path.exists(DB_PATH):
        return json.load(open(DB_PATH))
    return {}


def save_db(db):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    tmp = DB_PATH + ".tmp"
    json.dump(db, open(tmp, "w"), indent=2, sort_keys=True)
    os.replace(tmp, DB_PATH)


def boot_log():
    return run(["journalctl", "-k", "-b", "--no-pager", "-o", "cat"], check=False).stdout


def check_driver():
    """The fencing patch must be loaded, and every static BAR1 must start at offset 0."""
    offs = re.findall(r"CMP_BADBLK: static BAR1 enabled=(\d) startOffset=(0x[0-9a-f]+)", boot_log())
    if not offs:
        sys.exit("No CMP_BADBLK lines in this boot's kernel log: the driver with cmp-badblk.patch is not loaded.")
    bad = [o for o in offs if o != ("1", "0x0")]
    if bad:
        sys.exit(f"Static BAR1 is not at offset 0 on every card ({bad}); BAR1 offsets would not equal "
                 "physical addresses. Stopping.")


def ensure_badmap():
    os.makedirs(STATE_DIR, exist_ok=True)
    src, binp = os.path.join(TOOL_DIR, "badmap.cu"), os.path.join(STATE_DIR, "badmap")
    if os.path.exists(binp) and os.path.getmtime(binp) >= os.path.getmtime(src):
        return binp
    print("building badmap with", IMG_DEVEL, flush=True)
    shutil.copy(src, os.path.join(STATE_DIR, "badmap.cu"))
    run(["docker", "run", "--rm", "-v", f"{STATE_DIR}:/w", "-w", "/w", IMG_DEVEL,
         "nvcc", "-O2", "-arch=sm_80", "-o", "badmap", "badmap.cu"])
    return binp


def bar1_map(bdf):
    """mmap the full BAR1 read-only. The sysfs file keeps its boot-time size after the driver
    resizes BAR1, so take the size from the resource table and map through libc."""
    start, end = [int(x, 16) for x in open(f"/sys/bus/pci/devices/{bdf}/resource").read().split("\n")[1].split()[:2]]
    size = end - start + 1
    fd = os.open(f"/sys/bus/pci/devices/{bdf}/resource1", os.O_RDONLY | os.O_SYNC)
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    addr = libc.mmap(None, size, 1, 1, fd, 0)         # PROT_READ, MAP_SHARED
    if addr in (None, ctypes.c_void_p(-1).value):
        sys.exit(f"mmap of BAR1 failed for {bdf}: errno {ctypes.get_errno()}")
    return (ctypes.c_char * size).from_address(addr), size


def scan_tags(bdf):
    m, size = bar1_map(bdf)
    found = {}
    for off in range(0, size, BLOCK):
        a, b, c, d = struct.unpack_from("<4I", m, off)
        if a == MAGIC0 and c == MAGIC1 and d == (~b & 0xFFFFFFFF):
            found[b] = off
    return found


def map_card(card, passes, binp):
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    out = os.path.join(STATE_DIR, "runs", f"{stamp}-{card['uuid']}")
    os.makedirs(out)
    name = f"cmp-badblk-{card['uuid'][4:12]}"
    print(f"\n== GPU {card['index']} {card['pci']} {card['uuid']}: {passes} passes", flush=True)
    run(["docker", "run", "-d", "--rm", "--name", name, "--runtime=nvidia", "--gpus", f"device={card['uuid']}",
         "-v", f"{STATE_DIR}:/w:ro", "-v", f"{out}:/o", IMG_RUNTIME,
         "sh", "-c", f"/w/badmap /o {passes} > /o/badmap.log 2>&1"])
    try:
        for _ in range(3600):
            if os.path.exists(os.path.join(out, "ready")):
                break
            if not run(["docker", "ps", "-q", "-f", f"name={name}"]).stdout.strip():
                sys.exit(f"badmap exited early, see {out}/badmap.log:\n" + open(os.path.join(out, "badmap.log")).read())
            time.sleep(1)
        else:
            sys.exit(f"badmap did not finish within an hour, see {out}/badmap.log")
        print(open(os.path.join(out, "badmap.log")).read().strip().splitlines()[-2], flush=True)
        tags = scan_tags(card["pci"])
    finally:
        open(os.path.join(out, "done"), "w").close()
    nblocks = sum(int(r["blocks"]) for r in csv.DictReader(open(os.path.join(out, "chunks.csv"))))
    if len(tags) != nblocks:
        sys.exit(f"BAR1 scan found {len(tags)} of {nblocks} tagged blocks; the mapping is incomplete. Stopping.")
    errs = list(csv.DictReader(open(os.path.join(out, "errors.csv"))))
    per_frame = collections.Counter(tags[int(e["block"])] >> 16 for e in errs)
    with open(os.path.join(out, "badframes.json"), "w") as f:
        json.dump({hex(k): v for k, v in sorted(per_frame.items())}, f, indent=2)
    print(f"   {len(errs)} bad words in {len(per_frame)} frames; all {nblocks} blocks mapped", flush=True)
    for fr, n in sorted(per_frame.items()):
        print(f"   frame 0x{fr << 16:x} (key value 0x{fr:x}): {n} errors")
    return sorted(per_frame), len(errs), out


def conf_text(db, present):
    parts = []
    for c in present:
        frames = db.get(c["uuid"], {}).get("frames", [])
        if frames:
            keys = ";".join(f"RmCmpBadBlk{i}=0x{int(fr, 16):x}" for i, fr in enumerate(frames))
            parts.append(f"pci={c['pci']};{keys}")
    if not parts:
        return None
    lines = ["# Written by cmp-badblk.py apply on " + datetime.datetime.now().isoformat(timespec="seconds"),
             "# Bad 64 KB HBM frames per CMP 170HX (FB physical >> 16), read by cmp-badblk.patch at boot."]
    for c in present:
        frames = db.get(c["uuid"], {}).get("frames", [])
        if frames:
            lines.append(f"#   {c['pci']} = {c['uuid']} (serial {c['serial']}): {len(frames)} frames")
    lines.append(f'options nvidia NVreg_RegistryDwordsPerDevice="{";".join(parts)}"')
    return "\n".join(lines) + "\n"


def boot_fences(present):
    """Frames the driver fenced in this boot, as {pci: set of frame values (physical >> 16)}.
    Newer patch builds log the PCI address; older ones only the driver's GPU number, which is
    then matched to the nvidia-smi index (an assumption, reported as such)."""
    fences, assumed = {}, False
    by_index = {c["index"]: c["pci"] for c in present}
    for line in boot_log().splitlines():
        # Only the per-card lines (static BAR1 and each blacklisted frame); the summary line
        # ("N frames blacklisted") carries no PCI address.
        if "CMP_BADBLK" not in line or not ("static BAR1" in line or "blacklist 64K" in line):
            continue
        m = re.search(r"pci=([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.0)", line)
        if m:
            pci = m.group(1)
        else:
            g = re.search(r"NVRM: GPU(\d+) ", line)
            if not g or int(g.group(1)) not in by_index:
                continue
            pci, assumed = by_index[int(g.group(1))], True
        fences.setdefault(pci, set())
        f = re.search(r"blacklist 64K frame at (0x[0-9a-f]+) status=0x0", line)
        if f:
            fences[pci].add(int(f.group(1), 16) >> 16)
    return fences, assumed


def wanted_fences(db, present):
    return {c["pci"]: set(int(x, 16) for x in db.get(c["uuid"], {}).get("frames", [])) for c in present}


def fence_state(db, present):
    """Print per card whether this boot's fences match the list; return True if all match."""
    have, assumed = boot_fences(present)
    want, ok = wanted_fences(db, present), True
    for c in present:
        h, w = have.get(c["pci"], set()), want[c["pci"]]
        state = "active" if h == w else "NOT active"
        ok &= h == w
        print(f"  GPU {c['index']} {c['pci']}: list {len(w)} frames, fenced this boot {len(h)} frames -> {state}")
    if assumed:
        print("  (this boot's log has no PCI addresses; driver GPU numbers were matched to nvidia-smi indexes)")
    return ok


def cmd_apply(_args):
    db, present = load_db(), cards()
    text = conf_text(db, present)
    old = open(CONF_PATH).read() if os.path.exists(CONF_PATH) else None
    if text is None:
        print("No card in the list has bad frames; nothing to write.")
        return
    for uuid in db:
        if not any(c["uuid"] == uuid for c in present):
            print(f"note: {uuid} is in {DB_PATH} but not in this machine; skipped")
    def body(t):
        return [l for l in (t or "").splitlines() if not l.startswith("#")]
    if body(old) == body(text):
        print(f"{CONF_PATH} already matches the cards' current addresses.")
    else:
        if old is not None:
            shutil.copy(CONF_PATH, CONF_PATH + ".bak-" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
        open(CONF_PATH, "w").write(text)
        print(f"wrote {CONF_PATH}:\n{text}")
    print("Fences in the running driver:")
    if fence_state(db, present):
        print("The running driver already fences every listed frame on the right card. No reboot needed.")
    else:
        print("COLD BOOT NEEDED for the driver to pick up the list (sudo shutdown -h now, then power on).")


def cmd_status(_args):
    db, present = load_db(), cards()
    print("CMP 170HX cards now:")
    for c in present:
        frames = db.get(c["uuid"], {}).get("frames", [])
        print(f"  GPU {c['index']} {c['pci']} {c['uuid']} serial {c['serial']}: {len(frames)} frames in list "
              f"{frames}")
    print(f"\n{CONF_PATH}:\n" + (open(CONF_PATH).read() if os.path.exists(CONF_PATH) else "  (missing)\n"))
    print("Fences in the running driver:")
    fence_state(db, present)
    print("\nThis boot's CMP_BADBLK log lines:")
    for line in boot_log().splitlines():
        if "CMP_BADBLK" in line:
            print("  " + line)


def services(action):
    if action == "stop":
        run(["docker", "compose", "stop", "llama-swap"], cwd=LLAMA_SWAP_DIR, check=False)
        ids = run(["docker", "ps", "-q", "-f", "name=^vllm-"]).stdout.split()
        if ids:
            run(["docker", "stop", "-t", "30", *ids], check=False)
        time.sleep(5)
    else:
        run(["docker", "compose", "up", "-d", "llama-swap"], cwd=LLAMA_SWAP_DIR, check=False)


def cmd_map(args):
    check_driver()
    binp = ensure_badmap()
    present = cards()
    if not present:
        sys.exit("No CMP 170HX found.")
    stopped = False
    if any(c["used_mib"] > 100 for c in present):
        if not args.stop_services:
            sys.exit("A 170HX has memory in use (llama-swap or vLLM?). Stop them, or rerun with --stop-services.")
        print("stopping llama-swap and vllm-* containers", flush=True)
        services("stop"); stopped = True
        present = cards()
        if any(c["used_mib"] > 100 for c in present):
            sys.exit("A 170HX still has memory in use after stopping services. Stopping.")
    db = load_db()
    try:
        for c in present:
            frames, nerr, out = map_card(c, args.passes, binp)
            entry = db.setdefault(c["uuid"], {"frames": []})
            merged = sorted(set(int(x, 16) for x in entry["frames"]) | set(frames))
            if len(merged) > MAX_KEYS:
                sys.exit(f"{c['uuid']} would need {len(merged)} frames; cmp-badblk.patch reads at most {MAX_KEYS}. "
                         "Nothing written.")
            new = len(merged) - len(entry["frames"])
            entry.update(frames=[f"0x{x:x}" for x in merged], serial=c["serial"], last_pci=c["pci"],
                         last_mapped=datetime.datetime.now().isoformat(timespec="seconds"),
                         last_errors=nerr, last_passes=args.passes, last_run=out)
            print(f"   list for this card: {len(merged)} frames ({new} new)")
        save_db(db)
    finally:
        if stopped:
            print("starting llama-swap", flush=True)
            services("start")
    print()
    cmd_apply(args)


def main():
    if os.geteuid() != 0:
        sys.exit("run with sudo")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map"); m.add_argument("--passes", type=int, default=500)
    m.add_argument("--stop-services", action="store_true", help="stop llama-swap and vllm-* containers while testing")
    sub.add_parser("apply"); sub.add_parser("status")
    args = ap.parse_args()
    {"map": cmd_map, "apply": cmd_apply, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
