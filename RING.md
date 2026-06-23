# RING.md — Refactor plan: loaders that never stop

## Why we're doing this

Two problems with how batches flow today:

1. **The hang.** When a loader finishes its file list, the trainer has nothing left to consume and ends up blocked forever waiting for a batch that will never come. The trainer also has no clean way to say "okay, give me another pass over the data".
2. **Two species, two clocks.** Human and mouse have different numbers of columns, so a "pass over all the data" finishes at different times for each. The current design assumes both finish together.

The fix in one sentence: **the loaders run forever, and the trainer decides when to stop.**

## The new mental model

Think of each loader as a faucet. You turn it on once, and from then on it just keeps producing batches — when it runs out of files, it reshuffles its own file list and starts over without telling anyone. It does not have a concept of "end of epoch" that the outside world has to react to.

The trainer is now the only thing that knows about progress. It does three things:

- Pulls batches and trains on them, exactly like today.
- Counts how many full passes the **primary lane** (human) has completed, and stops when that count hits a limit.
- Watches for "chunk just finished" markers on batches as they come by, and uses those as the moments to print loss.

The Ring is the messenger in the middle. It still owns the slot buffers and the producer/consumer protocol, but its job grows slightly: it has to carry a little extra metadata on each batch so the trainer can do its bookkeeping.

## Where "epoch" lives now

Today the word "epoch" lives in `main_gpu.cpp`. After this refactor:

- `main_gpu.cpp` does not know about epochs at all. It has a batch loop and a stop condition. Nothing else.
- Each lane inside the Ring carries an integer "pass counter" — how many times this lane has wrapped its file list. Lane 0 might be on pass 3 while lane 1 is on pass 5; that is fine and expected.
- The DataLoader, internally, also tracks the same counter so it can advance it when it reshuffles. The Ring's copy is what the trainer reads.

The trainer looks at the primary lane's pass counter to decide when to stop. When `pass_counter` for lane 0 reaches `--max-epochs N`, the trainer breaks out.

## How loss reporting works

Loss is reported **per lane, when that lane finishes a chunk.** A chunk here means the unit of work the chunk-loader thread reads from disk and hands to the batch-builder — a chunk usually produces many batches.

The flow:

- Each lane keeps its own running sum of loss and its own batch counter, reset to zero after every report.
- When the batch-builder finishes the last batch of a chunk, it sets a "this is the last batch of a chunk" marker on that batch's slot.
- The trainer, after training on a batch, checks the marker. If it's set, the trainer reads that lane's mean loss, prints it along with the lane id and pass number, and resets that lane's accumulator.

Why per-chunk and not every-N-batches: chunks are a natural unit the system already understands, the cadence scales with dataset size automatically (more files = more reports), and per-lane reporting falls out for free because each lane crosses chunk boundaries on its own schedule.

The GPU sync needed to read the mean loss is a small operation (microseconds to a few milliseconds). Producers are unaffected — and even if they were, the Ring's slot back-pressure means they'd just wait, not pile up.

## How training stops

Three steps, all in the trainer:

1. Before each `acquire_ready`, check whether the primary lane's pass counter has reached the limit. If yes, break.
2. Pull batch, train, release.
3. Loop.

This means the trainer might finish a partial pass on lane 0 — that's fine. We could also stop on the exact pass boundary, but the simpler "≥ limit, stop now" version is what we want first.

## Shutdown

When the trainer breaks out of its loop, it tells the Ring to stop. The Ring sets a shutdown flag and wakes everyone on the condvar. The loader threads (chunk-loader and batch-builder for each lane) notice the flag at their next wakeup, finish what they're doing, and exit. The Ring's destructor joins them.

This is new — today nothing tells loader threads to exit because they exit "naturally" when a pass ends. With auto-rolling loaders, "naturally" never happens.

## What changes, file by file (high level only)

- **`ring.h` / `ring.cu`** — Add per-lane pass counter. Add a per-slot "end of chunk" marker. Add a shutdown flag and a way to set it. `acquire_ready` no longer returns false to signal "epoch done"; it only returns false on shutdown. Remove or repurpose the existing `eof_after` flag; if we keep it, it now only means "end of chunk". Add a small API for the trainer to read a lane's current pass counter.
- **`gpu_data_loader.h` / `.cu`** — The chunk-loader thread, when it reaches the end of its file list, reshuffles (advancing the same RNG, no reseed) and continues. It increments the lane's pass counter at the moment it wraps. The batch-builder thread tags the last batch of each chunk with the chunk-end marker. `begin_epoch()` becomes `start()` — called once at construction time, never again. Threads exit on the Ring's shutdown flag.
- **`main_gpu.cpp`** — Remove the outer epoch loop. Keep one batch loop. Add `--max-epochs` and read the primary lane's pass counter to decide when to break. Move loss reporting from "after epoch" to "on chunk-end marker", per lane. Call the Ring's shutdown after the loop exits.
- **No changes** to the autoencoder, the kernels, the lognorm path, or the slot buffer mechanics.

## Small design points worth being explicit about

- **One primary lane.** The trainer needs to know which lane id is "primary" for termination purposes. Pass it on the command line; default 0.
- **Per-lane RNG keeps advancing.** No reseeding on wrap. This means run-to-run determinism is preserved as long as the initial seed is fixed. Two consecutive passes will see different orderings, as desired.
- **Lane independence.** Lanes do not coordinate epoch boundaries. Mouse pass 5 can begin while human is mid-pass-2. The Ring's alternation policy (round-robin, weighted, etc.) is orthogonal.
- **Back-pressure is free.** If the trainer pauses for any reason (loss print, future validation, checkpoint save), producers block in `acquire_free` automatically. Memory stays bounded to the prefetch depth.
- **Startup is unchanged.** First `acquire_ready` blocks until the first batch is published, same as today. No special warm-up needed.

## Open questions / not in scope

- **Validation, checkpoints, LR schedule.** None of these exist today. The refactor is structured so they can be added later as "report-time" actions, but we are not adding them now.
- **Stopping mid-pass vs at exact pass boundary.** Going with mid-pass for simplicity. If we want exact-boundary later, the trainer can spin until it sees the chunk-end marker on lane 0's last chunk.
- **Combined loss metric.** Per-lane only for now. A combined number can be added on top trivially if wanted later.
- **Weighted alternation.** Multi-species support requires choosing how often each lane is served. Tracked separately; this refactor assumes whatever policy the Ring already has continues to work.
