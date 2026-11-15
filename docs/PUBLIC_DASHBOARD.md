# Public dashboard (read-only)

One link for the submission that is **live while the station is on** and shows **the last data when it is off**.

```
visitor ─► Netlify page (permanent link)
              ├─ station online:  live data ─► Tailscale Funnel (https) ─► viewer :8772 (read-only) ─► control center :8770
              └─ station offline: last/ snapshot uploaded by ./sl publish --deploy
operator ─► control center :8770 (private, localhost only, full control)
```

| Piece | Where | What it does |
|---|---|---|
| Control center | `127.0.0.1:8770`, `secondlook-center` service | full control; **never** exposed |
| Viewer | `127.0.0.1:8772`, `secondlook-viewer` service, `secondlook/viewer.py` | forwards an allowlist of read-only requests, refuses every write (405), strips paths and pids, caches 1–10 s so visitors barely load the laptop, rate-limits per visitor, serves wrist-camera crops only (no table/side camera, no full frames) |
| Tunnel | Tailscale Funnel | stable public `https://<laptop>.<tailnet>.ts.net` → viewer |
| Page | Netlify (`site/`, built by `./sl publish`) | same dashboard in view-only mode: "View only · live" or "View only · offline · last data …" |

The viewer's own URL is also a complete live page, so the Funnel link works on its own too.

## One-time setup (≈15 min)

**1. Tailscale Funnel** (needs a free Tailscale account):
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up                      # log in with the link it prints
sudo tailscale funnel --bg 8772        # first time: open the link it prints to allow Funnel, then run it again
tailscale funnel status                # shows https://<laptop>.<tailnet>.ts.net
```
Put that URL in `secondlook/config.yaml` → `public.api_url`. Funnel restarts by itself after a reboot.
Check it from your phone on mobile data: the page shows "View only · live".

**2. Netlify** (needs a free Netlify account):
```bash
npx netlify-cli login                  # once, opens the browser
./sl publish --deploy                  # first time it asks to create a site: pick a name, e.g. secondlook-junction
```
Put the site name in `config.yaml` → `public.netlify_site`. Your submission link is `https://<name>.netlify.app`.
No CLI? `./sl publish`, then drag the `site/` folder onto https://app.netlify.com/drop.

**3. Keep the laptop awake during the event** (plugged in):
Settings → Power → Automatic suspend: off. Keep the lid open, or ignore the lid on AC:
`sudo sed -i 's/^#\?HandleLidSwitchExternalPower=.*/HandleLidSwitchExternalPower=ignore/' /etc/systemd/logind.conf`
(takes effect after a reboot).

## Every day

- Station running = live link, nothing to do.
- **Before closing the laptop:** `./sl publish --deploy`. The offline page then shows this data.
- Services: `systemctl --user status secondlook-center secondlook-viewer`.

## What visitors can and cannot see

Can: status, counts, the last brick's wrist-camera photo and heatmap, the activity feed, insights charts, brick
history with wrist photos, model names. Cannot: start/stop, labels, model switching, voice, logs, table or side
camera images, full wrist frames, file paths. Writes are refused by the viewer itself (tested in
`tests/test_viewer.py`), not only hidden in the page.

**Never point the tunnel at port 8770** (the control center): its buttons move the robot.
