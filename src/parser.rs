use tokenizer;

#[derive(PartialEq, Debug, Clone)]
pub enum AstNode {
    Leaf(tokenizer::Token),
    Array(Vec<AstNode>),
    Function(Vec<AstNode>),
}

// This doesn't look like idiomatic rust..
fn do_parse(tokens: &[tokenizer::Token], offset: usize) -> (usize, Vec<AstNode>) {
    let mut ast = Vec::new();
    let mut i = offset;
    while i < tokens.len() {
        match &tokens[i] {
            &tokenizer::Token::EndFunction => {
                return (i + 1, ast);
            }
            &tokenizer::Token::EndArray => {
                return (i + 1, ast);
            }
            &tokenizer::Token::BeginFunction => {
                let (new_i, tmp) = do_parse(tokens, i + 1);
                i = new_i;
                // Check that there was a matching end
                match &tokens[i - 1] {
                    &tokenizer::Token::EndFunction => {}
                    _ => panic!("syntax error"),
                }
                ast.push(AstNode::Function(tmp));
            }
            &tokenizer::Token::BeginArray => {
                let (new_i, tmp) = do_parse(tokens, i + 1);
                i = new_i;
                // Check that there was a matching end
                match &tokens[i - 1] {
                    &tokenizer::Token::EndArray => {}
                    _ => panic!("syntax error"),
                }
                ast.push(AstNode::Array(tmp));
            }
            token => {
                i += 1;
                // This doesn't really seem right
                let tmp = AstNode::Leaf((*token).clone());
                ast.push(tmp);
            }
        }
    }
    (i, ast)
}

pub fn parse(a: &[tokenizer::Token]) -> Vec<AstNode> {
    let (i, ast) = do_parse(a, 0);
    if i != a.len() {
        panic!("parse error");
    }
    ast
}

#[test]
#[should_panic]
fn test_syntax_error() {
    parse(&[tokenizer::Token::BeginFunction]);
    parse(&[tokenizer::Token::BeginArray]);
    parse(&[tokenizer::Token::EndFunction]);
    parse(&[tokenizer::Token::EndArray]);
    parse(&[tokenizer::Token::BeginFunction,
            tokenizer::Token::Integer(1),
            tokenizer::Token::BeginArray,
            tokenizer::Token::Integer(2),
            tokenizer::Token::Integer(3),
            tokenizer::Token::EndFunction,
            tokenizer::Token::EndArray]);
}

#[test]
fn test_parser() {
    assert_eq!(parse(&[tokenizer::Token::BeginArray,
                       tokenizer::Token::Integer(1),
                       tokenizer::Token::Integer(2),
                       tokenizer::Token::EndArray]),
               [AstNode::Array(vec![AstNode::Leaf(tokenizer::Token::Integer(1)),
                                    AstNode::Leaf(tokenizer::Token::Integer(2))])]);
    assert_eq!(parse(&[tokenizer::Token::BeginFunction,
                       tokenizer::Token::Integer(1),
                       tokenizer::Token::Integer(2),
                       tokenizer::Token::EndFunction]),
               [AstNode::Function(vec![AstNode::Leaf(tokenizer::Token::Integer(1)),
                                       AstNode::Leaf(tokenizer::Token::Integer(2))])]);
    assert_eq!(parse(&[tokenizer::Token::BeginFunction,
                       tokenizer::Token::Integer(1),
                       tokenizer::Token::BeginArray,
                       tokenizer::Token::Integer(2),
                       tokenizer::Token::Integer(3),
                       tokenizer::Token::EndArray,
                       tokenizer::Token::EndFunction]),
               [AstNode::Function(vec![AstNode::Leaf(tokenizer::Token::Integer(1)),
                                       AstNode::Array(vec![AstNode::Leaf(tokenizer::Token::Integer(2)),
                                                           AstNode::Leaf(tokenizer::Token::Integer(3))])])]);
}
