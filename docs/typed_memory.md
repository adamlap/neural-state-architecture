# NSA Typed Persistent Memory

Memory is treated as a state-bearing system boundary, not an untyped text cache.

Each item contains:

$$
M=(id,content,type,provenance,sensitivity,time,expiry)
$$

The first implementation is immutable and append-only. Reads can filter expired entries without mutating the store.

## Security properties

- memory retains provenance references;
- sensitivity is explicit metadata;
- duplicate identities are rejected;
- expiry is explicit;
- memory writes are distinct from model-generated claims;
- future policy must control which state dimensions may enter memory.

## Future architecture

```text
model claim
    |
    +--> provenance validation
    |
    +--> sensitivity classification
    |
    +--> flow policy
    |
    +--> memory capability
    |
    v
typed memory
```

This prevents the memory layer from becoming a privilege-escalation or provenance-erasure boundary.
