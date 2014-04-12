#![feature(phase)]
extern crate regexp;
#[phase(syntax)] extern crate regexp_re;

use regexp::Regexp;

#[deriving(Eq)]
#[deriving(Show)]
pub enum Token {
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

pub fn tokenize(text: &str) -> ~[Token] {
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

    return tokenlist;
}

#[cfg(test)]
mod tests {
    // TODO: surely there must be an easier way to import stuff?
    use super::tokenize;
    use super::Integer;
    use super::Real;
    use super::String;
    use super::Boolean;
    use super::Identifier;
    use super::Binder;
    use super::Operator;
    use super::BeginArray;
    use super::EndArray;
    use super::BeginFunction;
    use super::EndFunction;
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
        assert_eq!(tokenize("addi"), ~[Operator(~"addi")]);
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
}