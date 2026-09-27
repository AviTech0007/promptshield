# Likely judge questions and honest answers

**"Isn't this just a wrapper around Prompt Guard / an existing classifier?"**
No. The classifier is one of four signals. We add hidden-text extraction, similarity to known attacks, and an action gate that works even when detection fails. Our ablation table compares the classifier alone with the full system, on data it wasn't trained on.

**"Keyword rules are trivial to get around."**
Agreed. Rules exist so decisions can be explained, not to be the defence. Reworded attacks are what the similarity signal and the action gate handle. The demo shows the gate stopping an attack with every detector switched off.

**"You tested on attacks you wrote yourselves."**
Part of our test set was written by a teammate who never saw our rules (`data/team_test_blind.jsonl`). Public datasets and Microsoft's BIPIA benchmark are in the test split, and nothing from TEST is in the similarity library or used for training.

**"The taint check just compares strings."**
Yes, after undoing common obfuscation ("x at y dot example"). It's deliberately simple and hard to fool for destination arguments. Limits:

- data the agent rewords itself can't be tracked;
- actions with no destination (like "delete everything") are covered by bulk and risk rules instead.

**"Users will just click Approve on everything."**
We only ask about high-risk actions whose arguments came from untrusted content, and we show where the instruction came from. Report how many approvals per 100 emails your eval run produced.

**"Can't the LLM judge be prompt-injected too?"**
Yes, which is why:

- it only sees unclear cases;
- it must return strict JSON;
- invalid output counts as suspicious;
- it can move the score only halfway;
- it can never approve an action.

**"Prompt injection is an unsolved problem."**
Agreed. We reduce risk with layered defence; we don't claim to cure it. The idea behind our gate (tracking where data came from and restricting what it can trigger) follows published work such as Google DeepMind's CaMeL and Simon Willison's "dual LLM" pattern, in a much simpler form.

**"How is this different from LLM Guard or NeMo Guardrails?"**
Those focus on filtering text going into and out of the model. Our main addition is the taint-tracking action gate, packaged as a two-line SDK with a human approval queue and an audit log.

**"What about images, PDFs, memory poisoning?"**
They're out of scope and listed in the README's Limitations section.

**"Latency?"**
The fast signals take milliseconds to about 100 ms per email on a CPU. The LLM judge only runs for unclear sentences. The median and p95 are in the Evaluation tab.

**"Who wrote this code?"**
We did, with AI assistance, which is disclosed. Any of us can explain any module; ask us.

## Never claim

- "prevents all prompt injection"
- "99% accurate" (unless measured on data you didn't write)
- "production-ready"
