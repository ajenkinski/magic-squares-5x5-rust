use clap::Parser;
use flate2::write::GzEncoder;
use flate2::Compression;
use rayon::prelude::ParallelIterator;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::channel;
use std::thread;
use std::time::Instant;

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

    /// If given, squares will be written to this file
    #[clap(short, long)]
    out_file: Option<PathBuf>,

    /// Compress output file using gzip compression
    #[clap(short, long)]
    compress: bool,

    /// Stop after generating this many squares.
    #[clap(long)]
    max_squares: Option<usize>,
}

fn main() {
    let options = Options::parse();

    let max_squares = options.max_squares.unwrap_or(usize::MAX);

    let env = enumerate::Env::new();

    let mut out_file: Option<_> = options.out_file.map(|path| -> Box<dyn Write> {
        let f = File::create(path).unwrap();
        if options.compress {
            Box::new(BufWriter::new(GzEncoder::new(f, Compression::default())))
        } else {
            Box::new(BufWriter::new(f))
        }
    });

    let start_time = Instant::now();

    if !options.multi_threaded {
        let mut num_squares = 0;
        for square in enumerate::generate_all_squares(&env).take(max_squares) {
            if num_squares % 1000 == 0 {
                print!("Found {} squares\r", num_squares);
                io::stdout().flush().unwrap();
            }
            num_squares += 1;

            if let Some(f) = out_file.as_mut() {
                enumerate::write_square(&square, f).unwrap()
            }
        }
        println!("Total squares found: {}", num_squares);
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(options.num_threads.unwrap_or_else(num_cpus::get))
            .build()
            .unwrap();

        println!("Running with {} threads", pool.current_num_threads());

        // Have worker threads send squares over a channel as they're generated, for the main thread to count and
        // possibly save.
        let (sender, receiver) = channel();

        // generate squares in a background thread so foreground thread can report progress and optionally save squares
        let generator_handle = thread::spawn(move || {
            pool.install(|| {
                // ParallelIterator doesn't support take(), so use an explicit count to end processing once
                // max_squares squares have been found
                let num_squares = AtomicUsize::new(0);

                enumerate::generate_all_squares_parallel(&env).try_for_each_with(
                    sender,
                    |sender, square| {
                        if num_squares.fetch_add(1, Ordering::Relaxed) < max_squares {
                            sender.send(square).unwrap();
                            Some(())
                        } else {
                            // break out early if max_squares reached.
                            None
                        }
                    },
                );
            });
        });

        let mut square_num = 0usize;
        for square in receiver {
            if square_num % 1000 == 0 {
                print!("Found {} squares\r", square_num);
                io::stdout().flush().unwrap();
            }

            if let Some(f) = out_file.as_mut() {
                enumerate::write_square(&square, f).unwrap()
            }

            square_num += 1;
        }
        println!("Total squares found: {}", square_num);

        generator_handle.join().unwrap();
    }

    println!("Took {}", humantime::format_duration(start_time.elapsed()));

    // It's recommended to call flush on a BufWriter before it's dropped.  Drop will also flush, but errors during
    // dropping will be ignored, so an explicit call is preferred.
    if let Some(f) = out_file.as_mut() {
        f.flush().unwrap()
    }
}
