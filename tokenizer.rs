#![feature(phase)]
extern crate regexp;
#[phase(syntax)] extern crate regexp_re;

use regexp::Regexp;

enum Token {
    Whitespace,
    Comment,
    BeginFunction,
    EndFunction,
    BeginArray,
    EndArray,
    Identifier(~str),
    Operator(~str),
    Binder(~str),
    Boolean(bool),
    Real(f64),
    Integer(i64),
    String(~str)
}

struct Result(Token, uint, bool);

fn whitespace_tokenizer(a: &str) -> Option<Result> {
    static re: Regexp = re!(r"^\s+");
    match re.find(a) {
        Some((0, x)) => Some(Result(Whitespace, x, false)),
        _ => None
    }
}

fn comment_tokenizer(a: &str) -> Option<Result> {
    static re: Regexp = re!(r"^%.*(\n|$)");
    match re.find(a) {
        Some((0, x)) => Some(Result(Comment, x, false)),
        _ => None
    }
}

fn begin_function_tokenizer(a: &str) -> Option<Result> {
    static re: Regexp = re!(r"^\{");
    match re.find(a) {
        Some((0, x)) => Some(Result(BeginFunction, x, true)),
        _ => None
    }
}

fn end_function_tokenizer(a: &str) -> Option<Result> {
    static re: Regexp = re!(r"^\}");
    match re.find(a) {
        Some((0, x)) => Some(Result(EndFunction, x, true)),
        _ => None
    }
}

fn begin_array_tokenizer(a: &str) -> Option<Result> {
    static re: Regexp = re!(r"^\[");
    match re.find(a) {
        Some((0, x)) => Some(Result(BeginArray, x, true)),
        _ => None
    }
}

fn end_array_tokenizer(a: &str) -> Option<Result> {
    static re: Regexp = re!(r"^\]");
    match re.find(a) {
        Some((0, x)) => Some(Result(EndArray, x, true)),
        _ => None
    }
}

fn boolean_tokenizer(a: &str) -> Option<Result> {
    static re: Regexp = re!(r"^true|false");
    match re.find(a) {
        Some((0, x)) if a.slice(0, x) == "true" => Some(Result(Boolean(true), x, true)),
        Some((0, x)) if a.slice(0, x) == "false" => Some(Result(Boolean(false), x, true)),
        _ => None
    }
}

fn identifier_tokenizer(a: &str) -> Option<Result> {
    static re: Regexp = re!(r"^[a-zA-Z][a-zA-Z0-9-_]*");
    match re.find(a) {
        Some((0, x)) => Some(Result(Identifier(a.slice(0, x).to_owned()), x, true)),
        _ => None
    }
}

fn operator_tokenizer(a: &str) -> Option<Result> {
    static re: Regexp = re!(r#"^acos|addi|addf|apply|asin|clampf|cone|cos|cube|cylinder|difference|divi|divf|eqi|eqf|floor|frac|get|getx|gety|getz|if|intersect|length|lessi|lessf|light|modi|muli|mulf|negi|negf|plane|point|pointlight|real|render|rotatex|rotatey|rotatez|scale|sin|sphere|spotlight|sqrt|subi|subf|translate|union|uscale"#);
    match re.find(a) {
        Some((0, x)) => Some(Result(Operator(a.slice(0, x).to_owned()), x, true)),
        _ => None
    }
}

fn binder_tokenizer(a: &str) -> Option<Result> {
    static re: Regexp = re!(r"^/[a-zA-Z][a-zA-Z0-9-_]*");
    static reop: Regexp = re!(r#"^/(acos|addi|addf|apply|asin|clampf|cone|cos|cube|cylinder|difference|divi|divf|eqi|eqf|floor|frac|get|getx|gety|getz|if|intersect|length|lessi|lessf|light|modi|muli|mulf|negi|negf|plane|point|pointlight|real|render|rotatex|rotatey|rotatez|scale|sin|sphere|spotlight|sqrt|subi|subf|translate|union|uscale)"#);
    if reop.is_match(a) {
        return None // TODO: raise some sort of error?
    }
    match re.find(a) {
        Some((0, x)) => Some(Result(Binder(a.slice(1, x).to_owned()), x, true)),
        _ => None
    }
}

fn real_tokenizer(a: &str) -> Option<Result> {
    static re: Regexp = re!(r"^-{0,1}\d+(?:(?:\.\d+(?:[eE]-{0,1}\d+){0,1})|(?:[eE]-{0,1}\d+))");
    match re.find(a) {
        Some((0, x)) => Some(Result(Real(from_str::<f64>(a.slice(0, x)).unwrap()), x, true)),
        _ => None
    }
}

fn integer_tokenizer(a: &str) -> Option<Result> {
    static re: Regexp = re!(r"^-{0,1}\d+");
    match re.find(a) {
        Some((0, x)) => Some(Result(Integer(from_str::<i64>(a.slice(0, x)).unwrap()), x, true)),
        _ => None
    }
}

fn string_tokenizer(a: &str) -> Option<Result> {
    static re: Regexp = re!(r#"".*""#);
    match re.find(a) {
        Some((0, x)) => Some(Result(String(a.slice(1, x-1).to_owned()), x, true)),
        _ => None
    }
}

fn tokenize(text: &str) {
    let tokenizers = [whitespace_tokenizer,
                      comment_tokenizer,
                      begin_function_tokenizer,
                      end_function_tokenizer,
                      begin_array_tokenizer,
                      end_array_tokenizer,
                      boolean_tokenizer,
                      operator_tokenizer,
                      identifier_tokenizer,
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
    println!("{:?}", tokenlist);
}

fn main() {
    tokenize("1 % apa");
    tokenize("1 % apa\n2");
    tokenize("1");
    tokenize("123");
    tokenize("-1");
    tokenize("-123");
    tokenize("1 2");
    tokenize("123 321");
    tokenize("-1-1");
    tokenize("1.0");
    tokenize("-1.0");
    tokenize("1.0e12");
    tokenize("1e12");
    tokenize("1e-12");
    tokenize("1e-2");
    tokenize("\"test\"");
    tokenize("true");
    tokenize("false");
    tokenize("/x");
    tokenize("/x-y_2");
    tokenize("x");
    tokenize("x-y_2");
    tokenize("addi");
    tokenize("/addi");
    tokenize("[1 2]");
    tokenize("{1 2}");
    tokenize("{1 [2 3]}");
}

//     # Run some tests
//     test("1 % apa", [('Integer', 1)])
//     test("1 % apa\n2", [('Integer', 1), ('Integer', 2)])
//     test("1", [('Integer', 1)])
//     test("123", [('Integer', 123)])
//     test("-1", [('Integer', -1)])
//     test("-123", [('Integer', -123)])
//     test("1 2", [('Integer', 1), ('Integer', 2)])
//     test("123 321", [('Integer', 123), ('Integer', 321)])
//     test("-1-1", [('Integer', -1), ('Integer', -1)])
//     test("1.0", [('Real', 1.0)])
//     test("-1.0", [('Real', -1.0)])
//     test("1.0e12", [('Real', 1.0e12)])
//     test("1e12", [('Real', 1e12)])
//     test("1e-12", [('Real', 1e-12)])
//     test("\"test\"", [('String', 'test')])
//     test("true", [('Boolean', True)])
//     test("false", [('Boolean', False)])
//     test("/x", [('Binder', 'x')])
//     test("/x-y_2", [('Binder', 'x-y_2')])
//     test("x", [('Identifier', 'x')])
//     test("x-y_2", [('Identifier', 'x-y_2')])
//     test("addi", [('Operator', 'addi')])
//     test("[1 2]", [('BeginArray', None),
//                    ('Integer', 1),
//                    ('Integer', 2),
//                    ('EndArray', None)])
//     test("{1 2}", [('BeginFunction', None),
//                    ('Integer', 1),
//                    ('Integer', 2),
//                    ('EndFunction', None)])
//     test("{1 [2 3]}", [('BeginFunction', None),
//                        ('Integer', 1),
//                        ('BeginArray', None),
//                        ('Integer', 2),
//                        ('Integer', 3),
//                        ('EndArray', None),
//                        ('EndFunction', None)])

    


#[cfg(test)]
mod tests {
    #[test]
    fn return_none_if_empty() {
      // ... test code ...
    }
}