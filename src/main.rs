use clap::Parser;
use rayon;
use rayon::prelude::ParallelIterator;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::sync::mpsc::channel;
use std::thread;

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

    /// If given, path of file to write magic squares to
    #[clap(short, long)]
    out_file: Option<PathBuf>,
}

fn main() {
    let options = Options::parse();

    let env = enumerate::Env::new();

    let mut out_file = options
        .out_file
        .map(|path| BufWriter::new(File::create(path).unwrap()));

    if !options.multi_threaded {
        let mut num_squares = 0;
        for square in enumerate::generate_all_squares(&env) {
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
        if let Some(num_threads) = options.num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()
                .unwrap();
        }

        let mut num_squares: usize = 0;

        let (sender, receiver) = channel();

        thread::scope(|scope| {
            scope.spawn(|| {
                enumerate::generate_all_squares_parallel(&env).for_each_with(
                    sender,
                    |sender, square| {
                        sender.send(square).unwrap();
                    },
                );
            });

            for square in receiver.iter() {
                num_squares += 1;
                if num_squares % 1000 == 0 {
                    print!("Found {} squares\r", num_squares);
                    io::stdout().flush().unwrap();
                }

                if let Some(f) = out_file.as_mut() {
                    enumerate::write_square(&square, f).unwrap()
                } 
            }
        });

        println!("Total squares found: {}", num_squares);
    }

    out_file.as_mut().map(|f| f.flush().unwrap());
}
