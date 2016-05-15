extern crate core;
use self::core::str::FromStr;

#[derive(PartialEq, Debug, Clone)]
pub enum Ops {
    OpAcos,
    OpAddf,
    OpAddi,
    OpApply,
    OpAsin,
    OpClampf,
    OpCone,
    OpCos,
    OpCube,
    OpCylinder,
    OpDifference,
    OpDivf,
    OpDivi,
    OpEqf,
    OpEqi,
    OpFloor,
    OpFrac,
    OpGet,
    OpGetx,
    OpGety,
    OpGetz,
    OpIf,
    OpIntersect,
    OpLength,
    OpLessf,
    OpLessi,
    OpLight,
    OpModi,
    OpMulf,
    OpMuli,
    OpNegf,
    OpNegi,
    OpPlane,
    OpPoint,
    OpPointlight,
    OpReal,
    OpRender,
    OpRotatex,
    OpRotatey,
    OpRotatez,
    OpScale,
    OpSin,
    OpSphere,
    OpSpotlight,
    OpSqrt,
    OpSubf,
    OpSubi,
    OpTranslate,
    OpUnion,
    OpUscale,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Token {
    Whitespace,
    Comment,
    BeginFunction,
    EndFunction,
    BeginArray,
    EndArray,
    Identifier(String),
    Operator(Ops),
    Binder(String),
    Boolean(bool),
    Real(f64),
    Integer(i64),
    Str(String),
}

struct Result(Token, usize, bool);

fn whitespace_tokenizer(a: &str) -> Option<Result> {
    let mut consumed = 0;
    for (offset, char) in a.char_indices() {
        if char.is_whitespace() {
            consumed = offset + 1;
        } else {
            break;
        }
    }
    match consumed {
        0 => None,
        _ => Some(Result(Token::Whitespace, consumed, false)),
    }
}

fn comment_tokenizer(a: &str) -> Option<Result> {
    let mut consumed = 0;
    for (offset, char) in a.char_indices() {
        if offset == 0 && char != '%' {
            return None;
        }
        if char == '\n' {
            break;
        } else {
            consumed = offset + 1;
        }
    }
    match consumed {
        0 => None,
        _ => Some(Result(Token::Comment, consumed, false)),
    }
}

fn begin_function_tokenizer(a: &str) -> Option<Result> {
    if a.chars().next().unwrap() == '{' {
        Some(Result(Token::BeginFunction, 1, true))
    } else {
        None
    }
}

fn end_function_tokenizer(a: &str) -> Option<Result> {
    if a.chars().next().unwrap() == '}' {
        Some(Result(Token::EndFunction, 1, true))
    } else {
        None
    }
}

fn begin_array_tokenizer(a: &str) -> Option<Result> {
    if a.chars().next().unwrap() == '[' {
        Some(Result(Token::BeginArray, 1, true))
    } else {
        None
    }
}

fn end_array_tokenizer(a: &str) -> Option<Result> {
    if a.chars().next().unwrap() == ']' {
        Some(Result(Token::EndArray, 1, true))
    } else {
        None
    }
}

fn boolean_tokenizer(a: &str) -> Option<Result> {
    if a.starts_with("true") {
        Some(Result(Token::Boolean(true), 4, true))
    } else if a.starts_with("false") {
        Some(Result(Token::Boolean(false), 5, true))
    } else {
        None
    }
}

fn is_identifier_start(c: char) -> bool {
    match c {
        'a'...'z' => true,
        'A'...'Z' => true,
        _ => false,
    }
}

fn is_identifier_rest(c: char) -> bool {
    match c {
        '0'...'9' => true,
        'a'...'z' => true,
        'A'...'Z' => true,
        '-' => true,
        '_' => true,
        _ => false,
    }
}

fn is_decimal_number(c: char) -> bool {
    match c {
        '0'...'9' => true,
        _ => false,
    }
}

fn is_exponent(c: char) -> bool {
    match c {
        'e' => true,
        'E' => true,
        _ => false,
    }
}

fn match_operator(a: &str) -> Option<Ops> {
    match a {
        "acos" => Some(Ops::OpAcos),
        "addf" => Some(Ops::OpAddf),
        "addi" => Some(Ops::OpAddi),
        "apply" => Some(Ops::OpApply),
        "asin" => Some(Ops::OpAsin),
        "clampf" => Some(Ops::OpClampf),
        "cone" => Some(Ops::OpCone),
        "cos" => Some(Ops::OpCos),
        "cube" => Some(Ops::OpCube),
        "cylinder" => Some(Ops::OpCylinder),
        "difference" => Some(Ops::OpDifference),
        "divf" => Some(Ops::OpDivf),
        "divi" => Some(Ops::OpDivi),
        "eqf" => Some(Ops::OpEqf),
        "eqi" => Some(Ops::OpEqi),
        "floor" => Some(Ops::OpFloor),
        "frac" => Some(Ops::OpFrac),
        "get" => Some(Ops::OpGet),
        "getx" => Some(Ops::OpGetx),
        "gety" => Some(Ops::OpGety),
        "getz" => Some(Ops::OpGetz),
        "if" => Some(Ops::OpIf),
        "intersect" => Some(Ops::OpIntersect),
        "length" => Some(Ops::OpLength),
        "lessf" => Some(Ops::OpLessf),
        "lessi" => Some(Ops::OpLessi),
        "light" => Some(Ops::OpLight),
        "modi" => Some(Ops::OpModi),
        "mulf" => Some(Ops::OpMulf),
        "muli" => Some(Ops::OpMuli),
        "negf" => Some(Ops::OpNegf),
        "negi" => Some(Ops::OpNegi),
        "plane" => Some(Ops::OpPlane),
        "point" => Some(Ops::OpPoint),
        "pointlight" => Some(Ops::OpPointlight),
        "real" => Some(Ops::OpReal),
        "render" => Some(Ops::OpRender),
        "rotatex" => Some(Ops::OpRotatex),
        "rotatey" => Some(Ops::OpRotatey),
        "rotatez" => Some(Ops::OpRotatez),
        "scale" => Some(Ops::OpScale),
        "sin" => Some(Ops::OpSin),
        "sphere" => Some(Ops::OpSphere),
        "spotlight" => Some(Ops::OpSpotlight),
        "sqrt" => Some(Ops::OpSqrt),
        "subf" => Some(Ops::OpSubf),
        "subi" => Some(Ops::OpSubi),
        "translate" => Some(Ops::OpTranslate),
        "union" => Some(Ops::OpUnion),
        "uscale" => Some(Ops::OpUscale),
        _ => None,
    }
}

fn is_operator(a: &str) -> bool {
    match match_operator(a) {
        Some(_) => true,
        _ => false,
    }
}

fn match_identifier(a: &str) -> Option<&str> {
    if is_identifier_start(a.chars().next().unwrap()) {
        let mut consumed = 0;
        for (offset, char) in a.char_indices() {
            if is_identifier_rest(char) {
                consumed = offset + 1;
            } else {
                break;
            }
        }
        match consumed {
            0 => None,
            _ => Some(&a[0..consumed]),
        }
    } else {
        None
    }
}

fn identifier_tokenizer(a: &str) -> Option<Result> {
    match match_identifier(a) {
        Some(id) if !is_operator(id) => {
            Some(Result(Token::Identifier(id.to_string()), id.len(), true))
        }
        _ => None,
    }
}

fn operator_tokenizer(a: &str) -> Option<Result> {
    // Same as identifier, but with a reversed check for operator-ness
    match match_identifier(a) {
        Some(id) => {
            match match_operator(id) {
                Some(op) => Some(Result(Token::Operator(op), id.len(), true)),
                _ => None,
            }
        }
        _ => None,
    }
}

fn binder_tokenizer(a: &str) -> Option<Result> {
    if a.chars().next().unwrap() == '/' && a.len() > 1 {
        match match_identifier(&a[1..]) {
            Some(id) if !is_operator(id) => {
                Some(Result(Token::Binder(id.to_string()), id.len() + 1, true))
            }
            _ => None,  // TODO: maybe raise some error for binding to reserved word
        }
    } else {
        None
    }
}

fn eat_digits(a: &str) -> usize {
    let mut consumed = 0;
    for (offset, char) in a.char_indices() {
        if is_decimal_number(char) {
            consumed = offset + 1;
        } else {
            break;
        }
    }
    return consumed;
}

fn real_tokenizer(a: &str) -> Option<Result> {
    let mut consumed = 0;
    // Skip minus sign if any
    if a.chars().nth(consumed).unwrap() == '-' {
        consumed += 1;
        if a.len() == consumed {
            return None;
        }
    }
    // Eat digits
    let digits = eat_digits(&a[consumed..]);
    if digits == 0 {
        return None;
    }
    consumed += digits;
    if a.len() == consumed {
        return None;
    }
    // Then maybe decimals
    if a.chars().nth(consumed).unwrap() == '.' {
        consumed += 1;
        if a.len() == consumed {
            return None;
        }
        let decimals = eat_digits(&a[consumed..]);
        if decimals == 0 {
            return None;
        }
        consumed += decimals;
    } else {
        // If there's no decimal, there must be an exponent!
        if !is_exponent(a.chars().nth(consumed).unwrap()) {
            return None;
        }
    }
    // Then exponent
    if consumed < a.len() && is_exponent(a.chars().nth(consumed).unwrap()) {
        consumed += 1;
        if a.len() == consumed {
            return None;
        }
        // Skip minus sign if any
        if a.chars().nth(consumed).unwrap() == '-' {
            consumed += 1;
            if a.len() == consumed {
                return None;
            }
        }
        // Eat exponent digits
        let exponent_digits = eat_digits(&a[consumed..]);
        if exponent_digits == 0 {
            return None;
        }
        consumed += exponent_digits;
    }
    // If we've come this far, everything is a-ok
    return Some(Result(Token::Real(a[0..consumed].parse().unwrap()), consumed, true));
}

fn integer_tokenizer(a: &str) -> Option<Result> {
    if a.len() == 0 {
        return None;
    }
    let mut pos = 0;
    // Skip minus sign if any
    if a.chars().next().unwrap() == '-' {
        pos += 1;
    }
    let mut consumed = 0;
    for (offset, char) in a[pos..].char_indices() {
        if is_decimal_number(char) {
            consumed = offset + 1;
        } else {
            break;
        }
    }
    match consumed {
        0 => None,
        _ => {
            Some(Result(Token::Integer(a[0..(pos + consumed)].parse().unwrap()),
                        pos + consumed,
                        true))
        }
    }
}

fn string_tokenizer(a: &str) -> Option<Result> {
    if a.len() <= 1 {
        return None;
    }
    if a.chars().next().unwrap() == '"' && a.len() > 1 {
        let mut consumed = 0;
        for (offset, char) in a[1..].char_indices() {
            if char == '"' {
                break;
            } else {
                consumed = offset + 1;
            }
        }
        match consumed {
            0 => None,
            _ => {
                Some(Result(Token::Str(String::from_str(&a[1..(consumed + 1)]).unwrap()),
                            consumed + 2,
                            true))
            }  // Add 2 for the "'s
        }
    } else {
        None
    }
}

pub fn tokenize(text: &str) -> Vec<Token> {
    let tokenizers: [fn(&str) -> Option<Result>; 13] = [whitespace_tokenizer,
                                                        comment_tokenizer,
                                                        begin_function_tokenizer,
                                                        end_function_tokenizer,
                                                        begin_array_tokenizer,
                                                        end_array_tokenizer,
                                                        boolean_tokenizer,
                                                        identifier_tokenizer,
                                                        operator_tokenizer,
                                                        binder_tokenizer,
                                                        real_tokenizer,
                                                        integer_tokenizer,
                                                        string_tokenizer];

    let mut tokenlist = Vec::<Token>::new();
    let mut pos: usize = 0;
    loop {
        let last_pos = pos;
        for &tokenizer in tokenizers.iter() {
            match tokenizer(&text[pos..]) {
                Some(Result(token, consumed, emit)) => {
                    if emit {
                        tokenlist.push(token)
                    }
                    pos += consumed;
                    break;
                }
                None => {}
            }
        }
        if pos == last_pos || pos == text.len() {
            break;
        }
    }

    return tokenlist;
}

#[test]
fn test_comment_tokenizer_comment() {
    let result = comment_tokenizer("% blah");
    match result {
        None => assert!(false),
        Some(Result(token, consumed, emit)) => {
            assert_eq!(token, Token::Comment);
            assert_eq!(consumed, 6);
            assert_eq!(emit, false);
        }
    }
}

#[test]
fn test_comment_tokenizer_non_comment() {
    let result = comment_tokenizer("blah");
    match result {
        None => {}
        Some(_) => assert!(false),
    }
}

#[test]
fn test_tokenizer() {
    assert_eq!(tokenize("1 % apa"), [Token::Integer(1)]);
    assert_eq!(tokenize("1 % apa\n2"),
               [Token::Integer(1), Token::Integer(2)]);
    assert_eq!(tokenize("1"), [Token::Integer(1)]);
    assert_eq!(tokenize("123"), [Token::Integer(123)]);
    assert_eq!(tokenize("-1"), [Token::Integer(-1)]);
    assert_eq!(tokenize("-123"), [Token::Integer(-123)]);
    assert_eq!(tokenize("1 2"), [Token::Integer(1), Token::Integer(2)]);
    assert_eq!(tokenize("123 321"),
               [Token::Integer(123), Token::Integer(321)]);
    assert_eq!(tokenize("-1-1"), [Token::Integer(-1), Token::Integer(-1)]);
    assert_eq!(tokenize("1.0"), [Token::Real(1.0)]);
    assert_eq!(tokenize("-1.0"), [Token::Real(-1.0)]);
    assert_eq!(tokenize("1.0e12"), [Token::Real(1.0e12)]);
    assert_eq!(tokenize("1e12"), [Token::Real(1e12)]);
    assert_eq!(tokenize("1e-12"), [Token::Real(1e-12)]);
    assert_eq!(tokenize("\"test\""),
               [Token::Str(String::from_str("test").unwrap())]);
    assert_eq!(tokenize("true"), [Token::Boolean(true)]);
    assert_eq!(tokenize("false"), [Token::Boolean(false)]);
    assert_eq!(tokenize("/x"),
               [Token::Binder(String::from_str("x").unwrap())]);
    assert_eq!(tokenize("/x-y_2"),
               [Token::Binder(String::from_str("x-y_2").unwrap())]);
    assert_eq!(tokenize("x"),
               [Token::Identifier(String::from_str("x").unwrap())]);
    assert_eq!(tokenize("x-y_2"),
               [Token::Identifier(String::from_str("x-y_2").unwrap())]);
    assert_eq!(tokenize("addi"), [Token::Operator(Ops::OpAddi)]);
    assert_eq!(tokenize("addiblaj"),
               [Token::Identifier(String::from_str("addiblaj").unwrap())]);
    assert_eq!(tokenize("[1 2]"),
               [Token::BeginArray, Token::Integer(1), Token::Integer(2), Token::EndArray]);
    assert_eq!(tokenize("{1 2}"),
               [Token::BeginFunction, Token::Integer(1), Token::Integer(2), Token::EndFunction]);
    assert_eq!(tokenize("{1 [2 3]}"),
               [Token::BeginFunction,
                Token::Integer(1),
                Token::BeginArray,
                Token::Integer(2),
                Token::Integer(3),
                Token::EndArray,
                Token::EndFunction])
}
