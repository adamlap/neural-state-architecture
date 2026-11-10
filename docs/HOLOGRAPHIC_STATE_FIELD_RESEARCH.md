# Holographic State Field Research Note

**Status:** Exploratory v0.1; isolated under experiments/.

The HSF hypothesis is that associative memory can be represented by a fixed
multidimensional field whose cells contain distributed superpositions of
key/value patterns. Querying the field uses matched-filter correlation to
reconstruct a candidate value representation.

This is adjacent to distributed associative memory and holographic reduced
representations, but the initial implementation is intentionally simpler: a
2-D spatial key pattern is outer-product-bound with a channel value code and
superposed into a 3-D tensor. It should not be described as a new attention
mechanism until a language-model experiment demonstrates that it can replace
or approximate token interaction.

The first benchmark is a capacity/interference sweep. It reports accuracy and
storage together; fixed field size alone is not useful if retrieval quality
collapses. Exact retrieval is retained as a control, and conclusions must
distinguish synthetic associative recall from semantic reasoning.

See experiments/holographic_state_field/README.md for equations, commands,
limitations, and planned falsification tests.
