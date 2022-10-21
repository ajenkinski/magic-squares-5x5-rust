use std::io::{self, Write};

pub mod enumerate;

fn main() {
    let mut num_squares = 0;

    
    enumerate::generate_all_squares(&mut|_| {
        num_squares += 1;
        if num_squares % 1000 == 0 {
            print!("Found {} squares\r", num_squares);
            io::stdout().flush().unwrap();
        }
    });
    println!("Found {} total squares", num_squares);
}
