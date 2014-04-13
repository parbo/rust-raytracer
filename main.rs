mod evaluator;

fn main() {
    println!("ast: {:?}", evaluator::parser::parse(evaluator::parser::tokenizer::tokenize("{1 [2 3]}")));
}
