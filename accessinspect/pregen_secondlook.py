"""Pre-make (cache) every line SecondLook can speak in the Ava voice, so they play instantly and offline.
Run: .venv/bin/python pregen_secondlook.py   (needs internet; about 3 minutes; safe to re-run)"""
import asyncio
import sys

import speak_alert as sa

COLORS = [None, "red", "blue", "yellow", "green", "white", "orange", "black", "pink", "purple", "brown", "gray"]
NUM = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
SECONDLOOK = ["Good brick.", "Defect found.", "Not sure. Taking a second look.", "Missed it. Trying again.",
              "Ready.", "Hand me a brick.", "Stopped.", "The table is clear.", "Looking at the table.",
              "New weights loaded.", "The pink bowl.", "The blue bowl.", "That was the last one.",
              "Voice on.", "Voice off.", "I need help, please.", "Please wait a moment.", "Yes.", "No.",
              "Thank you.", "Can you check this brick, please?", "I am finished.", "Please say that again."] + [f"{n.capitalize()} left." for n in NUM[1:]]


def lines():
    out = list(SECONDLOOK)
    for c in COLORS:
        brick = f"{c} brick" if c else "brick"
        for key in ("pass", "reject"):
            out += [t.format(brick=brick) for t in sa.PHRASES[key] if c is None or "{brick}" in t]
        for d in sa.DEFECTS.values():
            out += [t.format(brick=brick, defect=d) for t in sa.PHRASES["reject_typed"] if c is None or "{brick}" in t]
    return list(dict.fromkeys(out))


async def main():
    todo = lines()
    sem = asyncio.Semaphore(6)
    failed = []

    async def one(t):
        async with sem:
            try:
                await sa.synth(t)
            except Exception as e:
                failed.append(t)
                print("FAILED:", t, e, flush=True)
    await asyncio.gather(*(one(t) for t in todo))
    print(f"cached {len(todo) - len(failed)}/{len(todo)} SecondLook lines", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
