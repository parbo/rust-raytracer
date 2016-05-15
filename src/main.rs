mod evaluator;
mod parser;
mod raytracer;
mod tokenizer;
mod vecmath;

#[cfg(not(test))]
fn main() {
    //    println!("ast: {:?}", parser::parse(tokenizer::tokenize("{1 [2 3]}")));
}
