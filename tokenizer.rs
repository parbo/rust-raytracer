extern crate std;

#[deriving(Eq, Show, Clone)]
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
    OpUscale
}

#[deriving(Eq, Show, Clone)]
pub enum Token {
    Whitespace,
    Comment,
    BeginFunction,
    EndFunction,
    BeginArray,
    EndArray,
    Identifier(~str),
    Operator(Ops),
    Binder(~str),
    Boolean(bool),
    Real(f64),
    Integer(i64),
    String(~str),
}

struct Result(Token, uint, bool);

fn whitespace_tokenizer(a: &str) -> Option<Result> {
    let mut consumed = 0;
    for (offset, char) in a.char_indices() {
        if std::char::is_whitespace(char) {
            consumed = offset + 1;
        } else {
            break;
        }
    }
    match consumed {
        0 => None,
        _ => Some(Result(Whitespace, consumed, false)),
    }
}

fn comment_tokenizer(a: &str) -> Option<Result> {
    if a.char_at(0) == '%' {
        let mut consumed = 0;
        for (offset, char) in a.char_indices() {
            if char == '\n' {
                break;
            } else {
                consumed = offset + 1;
            }
        }
        match consumed {
            0 => None,
            _ => Some(Result(Comment, consumed, false)),
        }
    } else {
        None
    }
}

fn begin_function_tokenizer(a: &str) -> Option<Result> {
    if a.char_at(0) == '{' {
        Some(Result(BeginFunction, 1, true))
    } else {
        None
    }
}

fn end_function_tokenizer(a: &str) -> Option<Result> {
    if a.char_at(0) == '}' {
        Some(Result(EndFunction, 1, true))
    } else {
        None
    }
}

fn begin_array_tokenizer(a: &str) -> Option<Result> {
    if a.char_at(0) == '[' {
        Some(Result(BeginArray, 1, true))
    } else {
        None
    }
}

fn end_array_tokenizer(a: &str) -> Option<Result> {
    if a.char_at(0) == ']' {
        Some(Result(EndArray, 1, true))
    } else {
        None
    }
}

fn boolean_tokenizer(a: &str) -> Option<Result> {
    if a.starts_with("true") {
        Some(Result(Boolean(true), 4, true))
    } else if a.starts_with("false") {
        Some(Result(Boolean(false), 5, true))
    } else {
        None
    }
}

fn is_identifier_start(c: char) -> bool {
    match c {
        'a' .. 'z' => true,
        'A' .. 'Z' => true,
        _ => false
    }
}

fn is_identifier_rest(c: char) -> bool {
    match c {
        '0' .. '9' => true,
        'a' .. 'z' => true,
        'A' .. 'Z' => true,
        '-' => true,
        '_' => true,
        _ => false
    }
}

fn is_decimal_number(c: char) -> bool {
    match c {
        '0' .. '9' => true,
        _ => false
    }
}

fn is_exponent(c: char) -> bool {
    match c {
        'e' => true,
        'E' => true,
        _ => false
    }
}

fn match_operator(a: &str) -> Option<Ops> {
    match a {
        "acos" => Some(OpAcos),
        "addf" => Some(OpAddf),
        "addi" => Some(OpAddi),
        "apply" => Some(OpApply),
        "asin" => Some(OpAsin),
        "clampf" => Some(OpClampf),
        "cone" => Some(OpCone),
        "cos" => Some(OpCos),
        "cube" => Some(OpCube),
        "cylinder" => Some(OpCylinder),
        "difference" => Some(OpDifference),
        "divf" => Some(OpDivf),
        "divi" => Some(OpDivi),
        "eqf" => Some(OpEqf),
        "eqi" => Some(OpEqi),
        "floor" => Some(OpFloor),
        "frac" => Some(OpFrac),
        "get" => Some(OpGet),
        "getx" => Some(OpGetx),
        "gety" => Some(OpGety),
        "getz" => Some(OpGetz),
        "if" => Some(OpIf),
        "intersect" => Some(OpIntersect),
        "length" => Some(OpLength),
        "lessf" => Some(OpLessf),
        "lessi" => Some(OpLessi),
        "light" => Some(OpLight),
        "modi" => Some(OpModi),
        "mulf" => Some(OpMulf),
        "muli" => Some(OpMuli),
        "negf" => Some(OpNegf),
        "negi" => Some(OpNegi),
        "plane" => Some(OpPlane),
        "point" => Some(OpPoint),
        "pointlight" => Some(OpPointlight),
        "real" => Some(OpReal),
        "render" => Some(OpRender),
        "rotatex" => Some(OpRotatex),
        "rotatey" => Some(OpRotatey),
        "rotatez" => Some(OpRotatez),
        "scale" => Some(OpScale),
        "sin" => Some(OpSin),
        "sphere" => Some(OpSphere),
        "spotlight" => Some(OpSpotlight),
        "sqrt" => Some(OpSqrt),
        "subf" => Some(OpSubf),
        "subi" => Some(OpSubi),
        "translate" => Some(OpTranslate),
        "union" => Some(OpUnion),
        "uscale" => Some(OpUscale),
        _ => None
    }
}

fn is_operator(a: &str) -> bool {
    match match_operator(a) {
        Some(_) => true,
        _ => false
    }
}

fn match_identifier<'a>(a: &'a str) -> Option<&'a str> {
    if is_identifier_start(a.char_at(0)) {
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
            _ => Some(a.slice(0, consumed))
        }
    } else {
        None
    }
}

fn identifier_tokenizer(a: &str) -> Option<Result> {
    match match_identifier(a) {
        Some(id) if !is_operator(id) => {
            Some(Result(Identifier(id.to_owned()), id.len(), true))
        }
        _ => None
    }
}

fn operator_tokenizer(a: &str) -> Option<Result> {
    // Same as identifier, but with a reversed check for operator-ness
    match match_identifier(a) {
        Some(id) => {
            match match_operator(id) {
                Some(op) => Some(Result(Operator(op), id.len(), true)),
                _ => None
            }
        }
        _ => None
    }
}

fn binder_tokenizer(a: &str) -> Option<Result> {
    if a.char_at(0) == '/' && a.len() > 1 {
        match match_identifier(a.slice_from(1)) {
            Some(id) if !is_operator(id) => {
                Some(Result(Binder(id.to_owned()), id.len() + 1, true))
            },
            _ => None  // TODO: maybe raise some error for binding to reserved word
        }
    } else {
        None
    }
}

fn eat_digits(a: &str) -> uint {
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
    if a.char_at(consumed) == '-' {
        consumed += 1;
        if a.len() == consumed {
            return None;
        }
    }
    // Eat digits
    let digits = eat_digits(a.slice_from(consumed));
    if digits == 0 {
        return None;
    }
    consumed += digits;
    if a.len() == consumed {
        return None;
    }
    // Then maybe decimals
    if a.char_at(consumed) == '.' {
        consumed += 1;
        if a.len() == consumed {
            return None;
        }
        let decimals = eat_digits(a.slice_from(consumed));
        if decimals == 0 {
            return None;
        }
        consumed += decimals;
    } else {
        // If there's no decimal, there must be an exponent!
        if !is_exponent(a.char_at(consumed)) {
            return None;
        }
    }
    // Then exponent
    if consumed < a.len() && is_exponent(a.char_at(consumed)) {
        consumed += 1;
        if a.len() == consumed {
            return None;
        }
        // Skip minus sign if any
        if a.char_at(consumed) == '-' {
            consumed += 1;
            if a.len() == consumed {
                return None;
            }
        }
        // Eat exponent digits
        let exponent_digits = eat_digits(a.slice_from(consumed));
        if exponent_digits == 0 {
            return None;
        }
        consumed += exponent_digits;
    }
    // If we've come this far, everything is a-ok
    return Some(Result(Real(from_str::<f64>(a.slice(0, consumed).to_owned()).unwrap()), consumed, true));
}

fn integer_tokenizer(a: &str) -> Option<Result> {
    let mut pos = 0;
    // Skip minus sign if any
    if a.char_at(pos) == '-' && a.len() > pos {
        pos += 1;
    }
    let mut consumed = 0;
    for (offset, char) in a.slice_from(pos).char_indices() {
        if is_decimal_number(char) {
            consumed = offset + 1;
        } else {
            break;
        }
    }
    match consumed {
        0 => None,
        _ => Some(Result(Integer(from_str::<i64>(a.slice(0, pos + consumed).to_owned()).unwrap()), pos + consumed, true))
    }
}

fn string_tokenizer(a: &str) -> Option<Result> {
    if a.char_at(0) == '"' && a.len() > 1 {
        let mut consumed = 0;
        for (offset, char) in a.slice_from(1).char_indices() {
            if char == '"' {
                break;
            } else {
                consumed = offset + 1;
            }
        }
        match consumed {
            0 => None,
            _ => Some(Result(String(a.slice(1, consumed + 1).to_owned()), consumed + 2, true)),  // Add 2 for the "'s
        }
    } else {
        None
    }
}

pub fn tokenize(text: &str) -> ~[Token] {
    let tokenizers = [whitespace_tokenizer,
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

    let mut tokenlist: ~[Token] = ~[];
    let mut pos: uint = 0;
    loop {
        let last_pos = pos;
        for &tokenizer in tokenizers.iter() {
            match tokenizer(text.slice(pos, text.len())) {
                Some(Result(token, consumed, emit)) => {
                    if emit {
                        tokenlist.push(token)
                    }
                    pos += consumed;
                    break;
                },
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
fn test_tokenizer() {
    assert_eq!(tokenize("1 % apa"), ~[Integer(1)]);
    assert_eq!(tokenize("1 % apa\n2"), ~[Integer(1), Integer(2)]);
    assert_eq!(tokenize("1"), ~[Integer(1)]);
    assert_eq!(tokenize("123"), ~[Integer(123)]);
    assert_eq!(tokenize("-1"), ~[Integer(-1)]);
    assert_eq!(tokenize("-123"), ~[Integer(-123)]);
    assert_eq!(tokenize("1 2"), ~[Integer( 1), Integer(2)]);
    assert_eq!(tokenize("123 321"), ~[Integer(123), Integer(321)]);
    assert_eq!(tokenize("-1-1"), ~[Integer(-1), Integer(-1)]);
    assert_eq!(tokenize("1.0"), ~[Real(1.0)]);
    assert_eq!(tokenize("-1.0"), ~[Real(-1.0)]);
    assert_eq!(tokenize("1.0e12"), ~[Real(1.0e12)]);
    assert_eq!(tokenize("1e12"), ~[Real(1e12)]);
    assert_eq!(tokenize("1e-12"), ~[Real(1e-12)]);
    assert_eq!(tokenize("\"test\""), ~[String(~"test")]);
    assert_eq!(tokenize("true"), ~[Boolean(true)]);
    assert_eq!(tokenize("false"), ~[Boolean(false)]);
    assert_eq!(tokenize("/x"), ~[Binder(~"x")]);
    assert_eq!(tokenize("/x-y_2"), ~[Binder(~"x-y_2")]);
    assert_eq!(tokenize("x"), ~[Identifier(~"x")]);
    assert_eq!(tokenize("x-y_2"), ~[Identifier(~"x-y_2")]);
    assert_eq!(tokenize("addi"), ~[Operator(OpAddi)]);
    assert_eq!(tokenize("addiblaj"), ~[Identifier(~"addiblaj")]);
    assert_eq!(tokenize("[1 2]"), ~[BeginArray,
                                    Integer(1),
                                    Integer(2),
                                    EndArray]);
    assert_eq!(tokenize("{1 2}"), ~[BeginFunction,
                                    Integer(1),
                                    Integer(2),
                                    EndFunction]);
    assert_eq!(tokenize("{1 [2 3]}"), ~[BeginFunction,
                                        Integer(1),
                                        BeginArray,
                                        Integer(2),
                                        Integer(3),
                                        EndArray,
                                        EndFunction])
}
