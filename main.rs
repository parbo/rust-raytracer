mod tokenizer;
mod parser;

fn main() {
    println!("ast: {:?}", parser::parse(tokenizer::tokenize("{1 [2 3]}")));
}
