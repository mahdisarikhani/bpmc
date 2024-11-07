# BPMC

A Rust-based project for analyzing and correcting errors in the Chialvo map.
- **Bethe-Peierls Approximation**: Estimates the number of dynamical solutions using belief propagation and population dynamics algorithms.
- **Monte Carlo Method**: Minimizes errors in parameters.

## Quick Start

### Bethe-Peierls Approximation

```console
$ cargo build --release --bin bp
$ ./target/release/bp N BETA [true|false]
```

### Monte Carlo Method

```console
$ cargo build --release --bin mc
$ ./target/release/mc [true|false]
```
