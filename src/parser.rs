use tokenizer;
use std::error::Error as StdError;
use std::fmt;

#[derive(PartialEq, Debug, Clone)]
pub enum AstNode {
    Leaf(tokenizer::Token),
    Array(Vec<AstNode>),
    Function(Vec<AstNode>),
}

#[derive(Debug)]
pub enum ParseError {
    SyntaxError,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        return f.write_str(self.description());
    }
}

impl StdError for ParseError {
    fn description(&self) -> &str {
        match *self {
            ParseError::SyntaxError => "SyntaxError",
        }
    }
}

// This doesn't look like idiomatic rust..
fn do_parse(tokens: &[tokenizer::Token], offset: usize) -> Result<(usize, Vec<AstNode>), ParseError> {
    let mut ast = Vec::new();
    let mut i = offset;
    while i < tokens.len() {
        match &tokens[i] {
            &tokenizer::Token::EndFunction => {
                return Ok((i + 1, ast));
            }
            &tokenizer::Token::EndArray => {
                return Ok((i + 1, ast));
            }
            &tokenizer::Token::BeginFunction => {
                let (new_i, tmp) = do_parse(tokens, i + 1)?;
                i = new_i;
                // Check that there was a matching end
                match &tokens[i - 1] {
                    &tokenizer::Token::EndFunction => {}
                    _ => return Err(ParseError::SyntaxError),
                }
                ast.push(AstNode::Function(tmp));
            }
            &tokenizer::Token::BeginArray => {
                let (new_i, tmp) = do_parse(tokens, i + 1)?;
                i = new_i;
                // Check that there was a matching end
                match &tokens[i - 1] {
                    &tokenizer::Token::EndArray => {}
                    _ => return Err(ParseError::SyntaxError),
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
    Ok((i, ast))
}

pub fn parse(a: &[tokenizer::Token]) -> Result<Vec<AstNode>, ParseError> {
    let (i, ast) = do_parse(a, 0)?;
    if i != a.len() {
        return Err(ParseError::SyntaxError);
    }
    Ok(ast)
}

#[test]
#[should_panic]
fn test_syntax_error() {
    parse(&[tokenizer::Token::BeginFunction]).expect("syntax error");
    parse(&[tokenizer::Token::BeginArray]).expect("syntax error");
    parse(&[tokenizer::Token::EndFunction]).expect("syntax error");
    parse(&[tokenizer::Token::EndArray]).expect("syntax error");
    parse(&[tokenizer::Token::BeginFunction,
            tokenizer::Token::Integer(1),
            tokenizer::Token::BeginArray,
            tokenizer::Token::Integer(2),
            tokenizer::Token::Integer(3),
            tokenizer::Token::EndFunction,
            tokenizer::Token::EndArray]).expect("syntax error");
}

#[test]
fn test_parser() {
    assert_eq!(parse(&[tokenizer::Token::BeginArray,
                       tokenizer::Token::Integer(1),
                       tokenizer::Token::Integer(2),
                       tokenizer::Token::EndArray]).unwrap(),
               [AstNode::Array(vec![AstNode::Leaf(tokenizer::Token::Integer(1)),
                                    AstNode::Leaf(tokenizer::Token::Integer(2))])]);
    assert_eq!(parse(&[tokenizer::Token::BeginFunction,
                       tokenizer::Token::Integer(1),
                       tokenizer::Token::Integer(2),
                       tokenizer::Token::EndFunction]).unwrap(),
               [AstNode::Function(vec![AstNode::Leaf(tokenizer::Token::Integer(1)),
                                       AstNode::Leaf(tokenizer::Token::Integer(2))])]);
    assert_eq!(parse(&[tokenizer::Token::BeginFunction,
                       tokenizer::Token::Integer(1),
                       tokenizer::Token::BeginArray,
                       tokenizer::Token::Integer(2),
                       tokenizer::Token::Integer(3),
                       tokenizer::Token::EndArray,
                       tokenizer::Token::EndFunction]).unwrap(),
               [AstNode::Function(vec![AstNode::Leaf(tokenizer::Token::Integer(1)),
                                       AstNode::Array(vec![AstNode::Leaf(tokenizer::Token::Integer(2)),
                                                           AstNode::Leaf(tokenizer::Token::Integer(3))])])]);
}
