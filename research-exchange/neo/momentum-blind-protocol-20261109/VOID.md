# Blind-round protocol note

This directory is a **post-hoc protocol disclosure**, not a valid blind verification.

Before Ray supplied the blind-test ordering rules, Neo had already:

1. opened `research-exchange/queen/answer-key-20261109/`;
2. copied the answer-key files beside the input pack;
3. run the supplied `reference_rebuild.py` and `verify_all.py`;
4. compared an independently written selector against the Queen answer key; and
5. committed the resulting unblinded outputs in commit `30dd71a8ad3c6bc44db6b026a709ad7e9e9a96fe`.

Therefore Neo cannot truthfully produce a fresh blind freeze from the same evidence. Any new selection code or output would be outcome-aware even if written from scratch.

## Status

- Blind annual Top1 round: **VOID / NOT RUN BLIND**
- Blind W10H2/W12H4 round: **VOID / NOT RUN BLIND**
- Existing commit `30dd71a...`: retain as **unblinded independent replay**, not as a blind result.
- Queen/GPT reconciliation: must not count Neo as an independent blind voter.

## Recovery options

A valid Neo blind test requires one of:

1. a new, previously unseen input vintage and hidden answer key;
2. a different date range held back from all prior packs;
3. a separate verifier with no access to this conversation or the answer-key files.

No research, freeze, sizing, or live authority is granted.
