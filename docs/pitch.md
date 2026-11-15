# SecondLook — pitch scripts (updated Sat: ACT pick, no VLA)

Fill every [bracket] from the results table. Never say a number we did not measure.

## Table demo (~130 words, ~60 s spoken; the robot cycle fills the rest)

> "Draw anything on top of this brick and put it in the outline." *(they draw; P1 presses GO when hands are clear)*
>
> "These are toddler bricks. A damaged one is a safety problem. Nobody programmed this arm's grasp. We showed it about [N] times with that second arm, and it learned to pick.
>
> Now watch the screen. The camera on the gripper sees the brick close up. This model only ever saw good bricks, so your drawing lights up on the heatmap. Defective. If it's unsure, it puts the brick back, grabs it again and takes a second look.
>
> If the gripper closes on nothing, it knows it missed and tries again.
>
> Everything runs on this laptop: picking on the graphics chip, inspection on the AI chip, [X] watts extra, no cloud. In our tests it sorted [Y]% correctly at [Z] seconds a brick. Questions?"

30-second version: draw, catch, "learned from demos, two-sided check, all on the laptop, [Y]%."

## Video (≤ 1:50, ~220 words voice-over)

| Time | Picture | Voice-over |
|---|---|---|
| 0:00–0:10 | Close-up: heatmap lights up on a drawn mark; brick drops in the red bowl | "This robot just caught a defect it has never seen before." |
| 0:10–0:25 | Toddler bricks on the plate; hands sorting by eye | "These bricks are for toddlers. Checking them by eye for hours is tiring, and tired eyes miss things." |
| 0:25–0:45 | Leader-arm teleop, sped up; training screen | "We taught the arm to pick by showing it, [N] times, in Intel's Physical AI Studio. It trained on the laptop's own graphics chip." |
| 0:45–1:15 | One full cycle at real speed, HUD visible: pick, inspection close-up, heatmap, bowl; then one uncertain brick re-grasped | "It picks a brick and holds it up to its own camera. A model trained only on good bricks scores it. Anything unusual lights up. Unsure? It grabs the brick again and takes a second look." |
| 1:15–1:35 | Device table and power graph | "Picking runs on the integrated GPU, inspection on the NPU, through OpenVINO. The whole loop adds [X] watts. Nothing goes to the cloud." |
| 1:35–1:50 | Results numbers, team, name | "[Y]% sorted correctly at [Z] seconds a brick. SecondLook: teach it once, and it never gets tired of looking." |

## Stage pitch (≤ 280 words with numbers filled; clip ≤ 15 s)

> *[Clip: drawn brick caught, dropped in the red bowl.]*
>
> "These bricks are made for toddlers. A cracked one ends up in a two-year-old's mouth. Today, a lot of that checking is a person looking at parts for hours, and [sourced number on miss rate].
>
> SecondLook automates it with a low-cost arm and a laptop. We taught the arm to pick by demonstration: about [N] examples with a second arm, in Intel's Physical AI Studio. It holds each brick up to the camera on its own gripper. A model trained only on good bricks scores it, and anything unusual lights up on this heatmap. Unsure? It puts the brick back, grabs it again and takes a second look. Still unsure? It goes to a human, not into a box.
>
> All of it runs on one Intel Core Ultra laptop: picking on the integrated GPU, inspection on the NPU, the CPU in charge. The whole loop adds [X] watts.
>
> We measured it. [Y]% of bricks sorted correctly at [Z] seconds each. The second look settled [e] uncertain calls.
>
> Getting there meant getting a brand-new chip to train at all. We traced a GPU crash down to one attention kernel, worked around it, and reported it to Intel.
>
> For a factory, this means the person who knows the part teaches the robot in an afternoon, and the data never leaves the building.
>
> SecondLook. Teach it once, and it never gets tired of looking."

## Five likely judge questions (≤ 20 s each)

1. **What is learned, what is scripted?** The pick is learned (ACT from [N] demos). Inspection and bowl positions are taught poses, because they never change. The judgement is learned (anomaly model).
2. **Why anomaly detection?** Defects are rare and unpredictable. We only need good examples. Your drawing proves it.
3. **What if it's unsure?** Second view first. Still unsure: defective bowl, human checks. It fails safe.
4. **Why the NPU?** Keeps the GPU free for the policy and adds only [W] watts.
5. **Who pays?** Small manufacturers doing manual visual checks; one cell per line instead of an integrator project per product.
