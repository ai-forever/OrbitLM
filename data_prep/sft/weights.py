"""Sampling weights for one SFT version.

Every weight this module returns is a sampling weight: the trainer draws
packed blocks with probability proportional to it, so

    epochs = (weight / sum_weights) * token_budget / packed_tokens.

A SmolLM3 recipe weight is not that. It is the fraction of a split kept in
the mixture, and every kept example is then seen the same number of times.
`recipe_sampling_weights` turns that fraction into a sampling weight,
`recipe_weight * packed_tokens`, so two splits with the same recipe weight
get the same number of epochs however large they are.

A version with `parts` gives each part its own fraction of the token budget.
Inside a part, recipe streams use `recipe_weight * packed_tokens` and packed
streams share the slice in proportion to packed tokens. `epoch_cap` does not
shrink the fraction; an overrun is reported.

A version without `parts` has one capped slice instead: streams marked
weight_mode epoch_cap share `additions_fraction` of the budget, shrunk when
the cap would otherwise push any of those streams past `max_epochs`.
"""


def recipe_sampling_weights(recipe_weights, packed_tokens):
    """Convert SmolLM3 mixture fractions into sampling weights.

    recipe_weights: {stream: fraction of that split kept in one pass}.
    packed_tokens: {stream: tokens in the packed stream}. A missing stream
    counts as empty and gets weight 0.
    """
    return {
        name: float(recipe) * float(packed_tokens.get(name, 0))
        for name, recipe in recipe_weights.items()
    }


def part_weights(parts, streams, packed_tokens, token_budget):
    """Sampling weights for a version split into parts.

    parts: [{id, fraction, weight_mode, epoch_cap}]. Fractions sum to 1.
    weight_mode recipe uses each stream's recipe weight times packed tokens.
    weight_mode packed splits the part in proportion to packed tokens.
    streams: [{id, part, weight}]. weight is required for recipe parts.
    Returns (weights, exceeded). exceeded lists streams whose epochs pass
    the part's epoch_cap. The fraction is not reduced.
    """
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    by_id = {}
    fraction_sum = 0.0
    for part in parts:
        part_id = part["id"]
        if part_id in by_id:
            raise ValueError(f"duplicate part {part_id}")
        mode = part["weight_mode"]
        if mode not in ("recipe", "packed"):
            raise ValueError(f"part {part_id} has weight_mode {mode!r}")
        fraction = float(part["fraction"])
        if fraction <= 0:
            raise ValueError(f"part {part_id} fraction must be positive")
        by_id[part_id] = {
            "fraction": fraction,
            "weight_mode": mode,
            "epoch_cap": float(part["epoch_cap"]),
            "streams": [],
        }
        fraction_sum += fraction
    if abs(fraction_sum - 1.0) > 1e-6:
        raise ValueError(f"part fractions sum to {fraction_sum}, not 1")

    for stream in streams:
        part_id = stream.get("part")
        if part_id not in by_id:
            raise ValueError(f"stream {stream['id']} has no part {part_id!r}")
        by_id[part_id]["streams"].append(stream)

    weights = {}
    exceeded = []
    for part_id, part in by_id.items():
        mode = part["weight_mode"]
        raw = {}
        for stream in part["streams"]:
            name = stream["id"]
            packed = float(packed_tokens.get(name, 0) or 0)
            if packed <= 0:
                raw[name] = 0.0
                continue
            if mode == "recipe":
                if "weight" not in stream:
                    raise ValueError(f"recipe stream {name} has no weight")
                raw[name] = float(stream["weight"]) * packed
            else:
                raw[name] = packed
        raw_total = sum(raw.values())
        if raw_total <= 0:
            raise ValueError(f"part {part_id} has no packed tokens")
        scale = part["fraction"] / raw_total
        for name, value in raw.items():
            weights[name] = value * scale
        # epochs = weight / sum(weights) * budget / packed, and sum(weights) is 1
        # when every part is included. Report against that total.
        for stream in part["streams"]:
            name = stream["id"]
            packed = float(packed_tokens.get(name, 0) or 0)
            if packed <= 0 or weights[name] <= 0:
                continue
            epochs = weights[name] * float(token_budget) / packed
            cap = part["epoch_cap"]
            if epochs > cap + 1e-6:
                exceeded.append({
                    "part": part_id,
                    "stream": name,
                    "epochs": epochs,
                    "epoch_cap": cap,
                    "fraction": part["fraction"],
                    "fraction_at_cap": cap / epochs * part["fraction"],
                })
    return weights, exceeded


def addition_weights(explicit_weights, auto_tokens, token_budget, max_epochs, additions_fraction):
    """Sampling weights for the single epoch-capped slice of a version without parts.

    explicit_weights: {stream: sampling weight} for streams that are not
    capped. auto_tokens: {stream: packed tokens} for the capped streams.
    A stream with no packed tokens gets weight 0.
    """
    names = list(auto_tokens)
    base = sum(explicit_weights.values())
    total_tokens = sum(tokens for tokens in auto_tokens.values() if tokens > 0)
    if base <= 0 or total_tokens <= 0 or token_budget <= 0:
        return {name: 0.0 for name in names}

    # epochs = fraction * budget / total_tokens, same for every capped stream
    # because weights inside the slice are proportional to packed tokens.
    fraction = min(float(additions_fraction), max_epochs * total_tokens / float(token_budget))
    fraction = min(fraction, 0.5)
    slice_weight = fraction / (1.0 - fraction) * base
    return {
        name: (slice_weight * (tokens / total_tokens) if tokens > 0 else 0.0)
        for name, tokens in auto_tokens.items()
    }


def epochs_at_budget(weights, packed_tokens, token_budget):
    """How many times each stream is sampled over token_budget packed tokens."""
    total = sum(weights.values())
    if total <= 0 or token_budget <= 0:
        return {name: 0.0 for name in weights}
    epochs = {}
    for name, weight in weights.items():
        tokens = packed_tokens.get(name, 0)
        if tokens <= 0 or weight <= 0:
            epochs[name] = 0.0
        else:
            epochs[name] = (weight / total) * token_budget / tokens
    return epochs
