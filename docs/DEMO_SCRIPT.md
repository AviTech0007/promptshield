# 3-minute demo script

**Setup before you start:**

- dashboard open (`streamlit run dashboard/app.py`);
- sidebar set to brain **ollama** (or **simulated**, if 8.1 failed), detectors **on**, judge **off**;
- click "Reset demo";
- `python scripts/preflight.py --full` shows 0 failures;
- backup video on the desktop.

| Time | Do | Say |
|---|---|---|
| 0:00 | Slide 2 (the attack) | "AI agents read content written by strangers, and strangers can write instructions. This invoice email has a hidden line that tells the agent to forward 20 emails." |
| 0:20 | Click **Run WITHOUT PromptShield**, open e3 | "Here's a normal email agent doing a normal task: summarise my inbox and handle routine requests. It read the hidden text… and leaked N emails to the attacker. The user saw nothing." |
| 0:50 | Click **Run WITH PromptShield**, open e3 | "Same inbox, same agent, two lines of PromptShield. The email is quarantined. These four bars are our signals, and here is the hidden sentence, highlighted, with a plain-English reason." |
| 1:20 | Open e4 (HR forward) | "Genuine requests still work: forwarding to HR went through, because HR is a trusted contact." |
| 1:35 | Turn **Detectors off** in the sidebar, run WITH again, open e7, then the **Approvals** tab | "Now the worst case: every detector is off. E7 is a polite, reworded attack. The agent tries to forward mail, but the address came only from an untrusted email the user never mentioned, so the gate holds it for a human. Detection can fail; the gate still stops the damage." |
| 2:05 | **Try to break it** tab | "You can try your own attack here." (Paste a prepared one if time is short.) |
| 2:25 | **Evaluation** tab | "Measured on N held-out examples: F1 X, false alarms Y% on tricky genuine emails, attack success from 100% to Z%, and each layer adds something; here's the ablation, including the existing classifier alone." |
| 2:50 | Close | "PromptShield: open source, runs locally, two lines to add. We defend the agent's actions, not just its inputs." |

## Backup plans

| If this breaks | Do this |
|---|---|
| The real LLM doesn't fall for the attack live | Switch the sidebar to **simulated** and say so. |
| The dashboard crashes | Run `python -m demo_agent.run_demo` in the terminal. |
| Everything fails | Play the backup video. |

## Record the walkthrough video with the same script

- Keep it to 2–3 minutes.
- Show the terminal `pytest` run (60+ passed) for 5 seconds at the end.
