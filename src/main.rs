use std::io::{self, Write};
use clap::Parser;
use rayon;

pub mod enumerate;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Options {
    /// Run in multi-threaded mode
    #[clap(short, long)]
    multi_threaded: bool,

    /// Number of threads to use in multi-threaded mode.  Defaults to number of cores
    #[clap(short, long)]
    num_threads: Option<usize>
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
    } else {
        if let Some(num_threads) = options.num_threads {
            rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global().unwrap();
        }
        
        let num_squares = enumerate::generate_all_squares_parallel(&env);
        println!("Found {} total squares", num_squares);
    }
}
