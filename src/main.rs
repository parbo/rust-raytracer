mod evaluator;
mod parser;
mod primitives;
mod raytracer;
mod tokenizer;
mod vecmath;
mod transform;

#[cfg(not(test))]
fn main() {
    //    println!("ast: {:?}", parser::parse(tokenizer::tokenize("{1 [2 3]}")));
}
