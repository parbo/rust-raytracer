mod evaluator;
mod parser;
mod primitives;
mod raytracer;
mod tokenizer;
mod vecmath;
mod transform;
mod lights;

use std::env;
use std::fs::File;
use std::io::prelude::*;

fn main() {
    let args: Vec<String> = env::args().collect();

    let filename = &args[1];
    let mut f = File::open(filename).expect("file not found");
    let mut contents = String::new();
    f.read_to_string(&mut contents)
        .expect("something went wrong reading the file");
    evaluator::run(&contents);
}
