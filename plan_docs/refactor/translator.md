# Translator

Master model. Holds one `Autoencoder` per species over a shared layer core.

## Responsibilities

- Own the shared `Layer` stack (one copy, referenced by every species' autoencoder).
- Construct per-species `Autoencoder`s on demand.
- Route `forward` / `backward` / `update` calls to the correct species' autoencoder based on the batch's species name.
- Expose the per-species `Autoencoder` for callers that need it (loss readout, checkpoint, inspection).

Translator does **not** touch data loading, streams, or the ring. It only knows about models.

## Layer layout — sandwich

For every species, the full layer stack is:

```
[species encoder layers] → [shared encoder layers] → [shared decoder layers] → [species decoder layers]
```

- The **shared** middle is one `shared_ptr<Layer>` sequence, constructed once in the Translator, aliased into every species' autoencoder.
- The **species-specific** ends are constructed fresh per species; their weights are private to that species.
- Symmetry: species decoder mirrors species encoder dims in reverse; shared decoder mirrors shared encoder in reverse.

The species-specific encoder's first layer has input dim = that species' feature count (`m`). The species-specific decoder's last layer has output dim = same `m`. All other dims come from the constructor's neuron list.

## Constructor

```cpp
Translator(int n_shared_layers,
           int n_species_layers,
           const std::vector<int>& shared_dims,     // size n_shared_layers, inner dims of shared encoder half
           const std::vector<int>& species_dims,    // size n_species_layers, inner dims of species encoder half
           std::mt19937& rng);
```

- Constructs the shared encoder and shared decoder layers immediately. Shared decoder dims = reverse of shared encoder dims. These are `shared_ptr<Layer>` from creation.
- Does **not** create any `Autoencoder`s.
- Stores `rng` reference for later per-species init.

*(Exact dim-list shape is an implementation detail — implementer may prefer a single `neurons` list plus split index. The point is: shared layers are built here, once.)*

## `.species(name, feature_count)`

```cpp
void species(const std::string& name, int feature_count);
```

- Builds the species-specific encoder (input dim = `feature_count`) and species-specific decoder (output dim = `feature_count`).
- Concatenates: `[species_enc] + [shared_enc] + [shared_dec] + [species_dec]` as a `vector<shared_ptr<Layer>>`.
- Constructs an `Autoencoder` from that list using the existing `Autoencoder::init(dims, layers_in, rng)` overload.
- Stores in an internal `unordered_map<string, unique_ptr<Autoencoder>>`.
- Idempotency: calling twice with the same name is an error (throw / assert).

## Forward / backward / update

Batch carries its species name. Translator dispatches:

```cpp
void forward(const Batch& b, cudaStream_t stream);
void backward_and_step(const Batch& b, float lr, cudaStream_t stream);
```

Each method:
1. Look up `Autoencoder* ae = models_.at(b.species_name())`.
2. Delegate to `ae->forward(b.sparse_view(), stream)` / `ae->backward_and_step(...)`.

No cross-species locking is needed: shared layers are updated per batch, one batch at a time (the trainer is single-threaded on the GPU trainer stream).

## `.model(name)` and `.model(batch)`

```cpp
Autoencoder& model(const std::string& name);
Autoencoder& model(const Batch& b);          // = model(b.species_name())
```

Returns the per-species autoencoder for direct use (loss readout, save, etc.). Throws if unknown.

## What Translator does *not* do

- No data loading, no ring, no streams beyond passthrough.
- No global loss accumulation — each `Autoencoder` already tracks its own epoch loss; callers use `.model(name).read_epoch_loss(...)`.
- No epoch counting — that's the trainer loop's job.

## Testability

- Construct Translator + register two dummy species with tiny dims. Verify:
  - Shared layer object identity: `translator.model("a").layer(k_shared)` and `translator.model("b").layer(k_shared)` are the same `shared_ptr`.
  - Species-specific layers are *distinct* pointers.
  - `forward` on a `Batch` with unknown species throws cleanly.
