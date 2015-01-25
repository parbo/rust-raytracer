use tokenizer;

#[deriving(Eq, Show, Clone)]
pub enum AstNode {
    Leaf(tokenizer::Token),
    Array(Box<[AstNode]>),
    Function(Box<[AstNode]>)
}

// This doesn't look like idiomatic rust..
fn do_parse(tokens: &[tokenizer::Token], offset: uint) -> (uint, Box<[AstNode]>) {
    let mut ast: Box<[AstNode]> = Box::new([]);
    let mut i = offset;
    while i < tokens.len() {
        match &tokens[i] {
            &tokenizer::Token::EndFunction => {
                return (i + 1, ast);
            },
            &tokenizer::Token::EndArray => {
                return (i + 1, ast);
            },
            &tokenizer::Token::BeginFunction => {
                let (new_i, tmp) = do_parse(tokens, i + 1);
                i = new_i;
                // Check that there was a matching end
                match &tokens[i-1] {
                    &tokenizer::Token::EndFunction => {},
                    _ => panic!("syntax error")
                }
                ast.push(AstNode::Function(tmp));
            },
            &tokenizer::Token::BeginArray => {
                let (new_i, tmp) = do_parse(tokens, i + 1);
                i = new_i;
                // Check that there was a matching end
                match &tokens[i-1] {
                    &tokenizer::Token::EndArray => {},
                    _ => panic!("syntax error")
                }
                ast.push(AstNode::Array(tmp));
            },
            token => {
                i += 1;
                ast.push(AstNode::Leaf(token.clone()));
            }
        }
    }
    (i, ast)
}

pub fn parse(a: &[tokenizer::Token]) -> Box<[AstNode]> {
    println!("tokens: {:?}", a);
    let (i, ast) = do_parse(a, 0);
    if i != a.len() {
        panic!("parse error");
    }
    println!("ast: {:?}", ast);
    ast
}

#[test]
#[should_fail]
fn test_syntax_error() {
    parse([tokenizer::BeginFunction]);
    parse([tokenizer::BeginArray]);
    parse([tokenizer::EndFunction]);
    parse([tokenizer::EndArray]);
    parse([tokenizer::BeginFunction,
           tokenizer::Integer(1),
           tokenizer::BeginArray,
           tokenizer::Integer(2),
           tokenizer::Integer(3),
           tokenizer::EndFunction,
           tokenizer::EndArray]);
}

#[test]
fn test_parser() {
    assert_eq!(parse([tokenizer::BeginArray,
                      tokenizer::Integer(1),
                      tokenizer::Integer(2),
                      tokenizer::EndArray]),
               Box::new([Array(Box::new([Leaf(tokenizer::Integer(1)),
                                         Leaf(tokenizer::Integer(2))]))]));
    assert_eq!(parse([tokenizer::BeginFunction,
                      tokenizer::Integer(1),
                      tokenizer::Integer(2),
                      tokenizer::EndFunction]),
               Box::new([Function(Box::new([Leaf(tokenizer::Integer(1)),
                                            Leaf(tokenizer::Integer(2))]))]));
    assert_eq!(parse([tokenizer::BeginFunction,
                      tokenizer::Integer(1),
                      tokenizer::BeginArray,
                      tokenizer::Integer(2),
                      tokenizer::Integer(3),
                      tokenizer::EndArray,
                      tokenizer::EndFunction]),
               Box::new([Function(Box::new([Leaf(tokenizer::Integer(1)),
                                            Array(Box::new([Leaf(tokenizer::Integer(2)),
                                                            Leaf(tokenizer::Integer(3))]))]))]))
}
