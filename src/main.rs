use std::io::{self, Write};

pub mod enumerate;

fn main() {
    let env = enumerate::Env::new();
    let use_parallel = true;
    if use_parallel {
        let mut num_squares = 0;
        for _ in enumerate::generate_all_squares(&env) {
            if num_squares % 1000 == 0 {
                print!("Found {} squares\r", num_squares);
                io::stdout().flush().unwrap();
            }
            num_squares += 1;
        }
    } else {
        let num_squares = enumerate::generate_all_squares_parallel(&env);
        println!("Found {} total squares", num_squares);
    }
}
