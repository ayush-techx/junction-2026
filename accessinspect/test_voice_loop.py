"""Check that announcements never block the robot loop. Run: python test_voice_loop.py"""
import time
from speak_alert import announce, wait_for_speech


def timed(event, **kw):
    t = time.perf_counter()
    announce(event, **kw)
    print(f"  -> announce() returned in {(time.perf_counter() - t) * 1000:.0f} ms")


t0 = time.perf_counter()
timed("start")
for _ in range(20):  # pretend the ACT control loop keeps ticking
    time.sleep(0.1)
print(f"control loop ran 20 ticks while speaking ({time.perf_counter() - t0:.1f}s)")
timed("pass")
timed("reject", defect="tape")
timed("complete", passed=1, rejected=1)
wait_for_speech()
print(f"all speech finished at {time.perf_counter() - t0:.1f}s")
