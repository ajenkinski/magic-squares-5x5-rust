use clap::Parser;
use rayon;
use rayon::prelude::ParallelIterator;
use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::{thread, time};

pub mod enumerate;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Options {
    /// Run in multi-threaded mode
    #[clap(short, long)]
    multi_threaded: bool,

    /// Number of threads to use in multi-threaded mode.  Defaults to number of cores
    #[clap(short, long)]
    num_threads: Option<usize>,
}

fn main() {
    let options = Options::parse();

    let env = enumerate::Env::new();

    if !options.multi_threaded {
        let mut num_squares = 0;
        for _ in enumerate::generate_all_squares(&env) {
            if num_squares % 1000 == 0 {
                print!("Found {} squares\r", num_squares);
                io::stdout().flush().unwrap();
            }
            num_squares += 1;
        }
        println!("Total squares found: {}", num_squares);
    } else {
        if let Some(num_threads) = options.num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()
                .unwrap();
        }

        thread::scope(|scope| {
            // So that I can print a running total, I have the worker threads increment num_squares
            // for each generated square, and then the main thread prints the current total every second.
            let num_squares = Arc::new(AtomicUsize::new(0));
            let num_squares_clone = num_squares.clone();

            let worker_thread = scope.spawn(move || {
                enumerate::generate_all_squares_parallel(&env).for_each(|_| {
                    num_squares_clone.fetch_add(1, Ordering::Relaxed);
                });
            });

            let poll_interval = time::Duration::from_secs(1);
            while !worker_thread.is_finished() {
                print!("Found {} squares\r", num_squares.load(Ordering::Relaxed));
                io::stdout().flush().unwrap();

                thread::sleep(poll_interval);
            }
            println!(
                "Total squares found: {}",
                num_squares.load(Ordering::Relaxed)
            );
        });
    }
}
