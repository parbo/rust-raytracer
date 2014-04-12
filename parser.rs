use tokenizer::Token;
use tokenizer::Program;

mod tokenizer;

#[deriving(Eq)]
#[deriving(Show)]
pub enum AstNode {
    Leaf(Token),
    Cons(Token, ~[~AstNode]),
    Nil
}

pub fn parse(a: ~[Token]) -> ~AstNode {
    println!("tokens: {:?}", a);
    ~Cons(Program, ~[~Nil])
}

#[cfg(test)]
mod tests {
    use super::parse;
    use super::Leaf;
    use super::Cons;
    use tokenizer::Program;
    use tokenizer::Array;
    use tokenizer::Function;
    use tokenizer::Integer;
    use tokenizer::BeginArray;
    use tokenizer::EndArray;
    use tokenizer::BeginFunction;
    use tokenizer::EndFunction;
    #[test]
    #[should_fail]
    fn test_syntax_error() {
        parse(~[BeginFunction]);
        parse(~[BeginArray]);
        parse(~[EndFunction]);
        parse(~[EndArray]);
        parse(~[BeginFunction,
                Integer(1),
                BeginArray,
                Integer(2),
                Integer(3),
                EndFunction,
                EndArray]);
    }

    #[test]
    fn test_parser() {
        assert_eq!(parse(~[BeginArray,
                           Integer(1),
                           Integer(2),
                           EndArray]),
                   ~Cons(Program, ~[~Cons(Array, ~[~Leaf(Integer(1)),
                                                   ~Leaf(Integer(2))])]));
        assert_eq!(parse(~[BeginFunction,
                           Integer(1),
                           Integer(2),
                           EndFunction]),
                   ~Cons(Program, ~[~Cons(Function, ~[~Leaf(Integer(1)),
                                                      ~Leaf(Integer(2))])]));
        assert_eq!(parse(~[BeginFunction,
                           Integer(1),
                           BeginArray,
                           Integer(2),
                           Integer(3),
                           EndArray,
                           EndFunction]),
                   ~Cons(Program, ~[~Cons(Function, ~[~Leaf(Integer(1)),
                                                      ~Cons(Array, ~[~Leaf(Integer(2)),
                                                                     ~Leaf(Integer(3))])])]))
    }
}