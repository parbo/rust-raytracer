mod evaluator;

#[cfg(not(test))]
fn main() {
    println!("ast: {:?}", evaluator::parser::parse(evaluator::parser::tokenizer::tokenize("{1 [2 3]}")));
}
