# NSA Capability & Authority

NSA treats authority as an externally issued, explicitly scoped capability rather than a property that a model can infer about itself.

A capability is:

$$
C=(id, issuer, subject, action, scope, purpose, expiry, nonce)
$$

An action is permitted only when the trusted authority can validate:

$$
C.subject=s
\land C.action=a
\land C.scope=q
\land C.expiry>t
$$

## Security principle

Model cognition may propose an action. It cannot mint the capability required to execute that action.

```text
model proposal
      |
      v
flow policy
      |
      v
trusted capability authority
      |
      v
validated action
```

## Least authority

Capabilities should be as narrow as possible in action and scope. A capability for `filesystem.read:/safe/data` must not imply `filesystem.write:/safe/data` or access to another path.

## Relationship to canonical state

Hard state may summarize trusted authorization facts, but the capability object remains the concrete authorization artifact. This avoids treating a neural representation of permission as permission itself.

## Next extensions

- revocation
- delegation with attenuation
- capability chains
- nonce/replay protection
- audit records
- resource quotas
- human approval gates
- tool gateway integration
