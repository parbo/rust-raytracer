use std::mem;

pub mod tokenizer;

#[deriving(Eq, Show, Clone)]
pub enum AstNode {
    Leaf(~tokenizer::Token),
    Array(~[AstNode]),
    Function(~[AstNode])
}

// This doesn't look like idiomatic rust..
fn do_parse(tokens: &[~tokenizer::Token], offset: uint) -> (uint, ~[AstNode]) {
    let mut ast: ~[AstNode] = ~[];
    let mut i = offset;
    while i < tokens.len() {
        match &tokens[i] {
            &~tokenizer::EndFunction => {
                return (i + 1, ast);
            },
            &~tokenizer::EndArray => {
                return (i + 1, ast);
            },
            &~tokenizer::BeginFunction => {
                let (new_i, tmp) = do_parse(tokens, i + 1);
                i = new_i;
                ast.push(Function(tmp));
            },
            &~tokenizer::BeginArray => {
                let (new_i, tmp) = do_parse(tokens, i + 1);
                i = new_i;
                ast.push(Array(tmp));
            },
            token => {
                i += 1;
                ast.push(Leaf(token.clone()));
            }
        }
    }
    (i, ast)
}

pub fn parse(a: ~[~tokenizer::Token]) -> ~[AstNode] {
    println!("tokens: {:?}", a);
    let (i, ast) = do_parse(a, 0);
    if i != a.len() {
        fail!("parse error");
    }
    println!("ast: {:?}", ast);
    ast
}

// #[test]
// #[should_fail]
// fn test_syntax_error() {
//     parse(~[~tokenizer::BeginFunction]);
//     parse(~[~tokenizer::BeginArray]);
//     parse(~[~tokenizer::EndFunction]);
//     parse(~[~tokenizer::EndArray]);
//     parse(~[~tokenizer::BeginFunction,
//             ~tokenizer::Integer(1),
//             ~tokenizer::BeginArray,
//             ~tokenizer::Integer(2),
//             ~tokenizer::Integer(3),
//             ~tokenizer::EndFunction,
//             ~tokenizer::EndArray]);
// }

#[test]
fn test_parser() {
    assert_eq!(parse(~[~tokenizer::BeginArray,
                       ~tokenizer::Integer(1),
                       ~tokenizer::Integer(2),
                       ~tokenizer::EndArray]),
               ~[Array(~[Leaf(~tokenizer::Integer(1)),
                         Leaf(~tokenizer::Integer(2))])]);
    assert_eq!(parse(~[~tokenizer::BeginFunction,
                       ~tokenizer::Integer(1),
                       ~tokenizer::Integer(2),
                       ~tokenizer::EndFunction]),
               ~[Function(~[Leaf(~tokenizer::Integer(1)),
                            Leaf(~tokenizer::Integer(2))])]);
    assert_eq!(parse(~[~tokenizer::BeginFunction,
                       ~tokenizer::Integer(1),
                       ~tokenizer::BeginArray,
                       ~tokenizer::Integer(2),
                       ~tokenizer::Integer(3),
                       ~tokenizer::EndArray,
                       ~tokenizer::EndFunction]),
               ~[Function(~[Leaf(~tokenizer::Integer(1)),
                            Array(~[Leaf(~tokenizer::Integer(2)),
                                    Leaf(~tokenizer::Integer(3))])])])
}
