# Deploy

| File | What |
|---|---|
| `secondlook-center.service` | systemd user unit that keeps the control center on http://127.0.0.1:8770. Copy it to `~/.config/systemd/user/`, adjust `WorkingDirectory`, then `systemctl --user enable --now secondlook-center`. |
| `station_monitor/monitor.py` | passive station monitor (http://127.0.0.1:8090): logs arm/camera/Studio events and never opens a serial port or camera itself. Run from Studio's backend folder: `uv run --extra xpu python <repo>/deploy/station_monitor/monitor.py`. Logs go to `~/station_monitor/`. |
