# Generate all 5x5 magic squares

This project implements the algorithm described at 

https://www.researchgate.net/publication/294288450_Generation_of_all_magic_squares_of_order_5_and_interesting_patterns_finding

# Setup

Follow the instructions here to install the Rust toolchain:

https://www.rust-lang.org/tools/install

The install script will install the Rust tools under ~/.cargo, and should add something to your .bashrc to add
the tools to your path. If not, manually add `~/.cargo/bin` to your path.

# Running

To run single-threaded, run 

```
cargo run -r
```

To run in multi-thread-mode, run

```
cargo run -r -- -m
```

The app will default to using all available cores on the machine it's run on.  You can use `-n` to specify the number of threads, as in:

```
cargo run -r -- -m -n 16
```
