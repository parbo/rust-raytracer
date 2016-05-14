use std::collections;
use parser;
use tokenizer;

#[derive(PartialEq, Debug, Clone)]
pub enum Value {
    ValClosure(Box<Env>, Vec<parser::AstNode>), // The ast for the function
    ValArray(Vec<Value>),
    ValBoolean(bool),
    ValReal(f64),
    ValInteger(i64),
    ValString(String)
}

pub type Env = collections::HashMap<String, Value>;

fn eval_op<'a>(op: &tokenizer::Ops, env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    match op {
        &tokenizer::Ops::OpAddi => eval_addi(env, stack),
        &tokenizer::Ops::OpAddf => eval_addf(env, stack),
        &tokenizer::Ops::OpApply => eval_apply(env, stack),
        &tokenizer::Ops::OpAcos => eval_acos(env, stack),
        &tokenizer::Ops::OpAsin => eval_asin(env, stack),
        &tokenizer::Ops::OpClampf => eval_clampf(env, stack),
        &tokenizer::Ops::OpCos => eval_cos(env, stack),
        &tokenizer::Ops::OpDivf => eval_divf(env, stack),
        &tokenizer::Ops::OpDivi => eval_divi(env, stack),
        &tokenizer::Ops::OpEqi => eval_eqi(env, stack),
        &tokenizer::Ops::OpEqf => eval_eqf(env, stack),
        &tokenizer::Ops::OpFloor => eval_floor(env, stack),
        &tokenizer::Ops::OpFrac => eval_frac(env, stack),
        &tokenizer::Ops::OpGet => eval_get(env, stack),
        &tokenizer::Ops::OpIf => eval_if(env, stack),
        &tokenizer::Ops::OpLength => eval_length(env, stack),
        &tokenizer::Ops::OpLessi => eval_lessi(env, stack),
        &tokenizer::Ops::OpLessf => eval_lessf(env, stack),
        &tokenizer::Ops::OpModi => eval_modi(env, stack),
        &tokenizer::Ops::OpMuli => eval_muli(env, stack),
        &tokenizer::Ops::OpMulf => eval_mulf(env, stack),
        &tokenizer::Ops::OpNegi => eval_negi(env, stack),
        &tokenizer::Ops::OpNegf => eval_negf(env, stack),
        &tokenizer::Ops::OpReal => eval_real(env, stack),
        &tokenizer::Ops::OpSin => eval_sin(env, stack),
        &tokenizer::Ops::OpSqrt => eval_sqrt(env, stack),
        &tokenizer::Ops::OpSubi => eval_subi(env, stack),
        &tokenizer::Ops::OpSubf => eval_subf(env, stack),
        _ => panic!("operator {:?} not implemented yet!", op)
    }
}


#[derive(PartialEq, Debug, Clone)]
pub enum Stack {
    Cons(Value, Box<Stack>),
    Nil
}

fn divi(a: i64, b: i64) -> i64 {
    ((a as f64) / (b as f64)).round() as i64
}

fn modi(a: i64, b: i64) -> i64 {
    a - b * divi(a, b)
}

fn get_integer(v: &Value) -> i64 {
    match v {
        &Value::ValInteger(i) => i,
        other => panic!("{:?} is not an integer", other)
    }
}

fn get_string<'a>(v: &'a Value) -> &'a String {
    match v {
        &Value::ValString(ref i) => &i,
        other => panic!("{:?} is not an integer", other)
    }
}

fn get_real(v: &Value) -> f64 {
    match v {
        &Value::ValReal(f) => f,
        other => panic!("{:?} is not an real", other)
    }
}

fn get_boolean(v: &Value) -> bool {
    match v {
        &Value::ValBoolean(b) => b,
        other => panic!("{:?} is not an boolean", other)
    }
}

fn make_array(s: Box<Stack>) -> Value {
    let mut tmp = Vec::new();
    let mut stack = s.clone();
    while !is_empty(&stack) {
        let (env, popped_stack) = pop(stack);
        stack = popped_stack;
        tmp.insert(0, env);
    }
    Value::ValArray(tmp)
}

// def make_point(x, y, z):
//     return ('Point', (x,y,z))

// def get_point(p):
//     assert check_point(p)
//     return get_value(p)

// def get_point_x(p):
//     point = get_point(p)
//     return point[0]

// def get_point_y(p):
//     point = get_point(p)
//     return point[1]

// def get_point_z(p):
//     point = get_point(p)
//     return point[2]

// def get_node(obj):
//     if not isinstance(obj, primitives.Node):
//         raise GMLTypeError
//     return obj

fn push(stack: Box<Stack>, value: Value) -> Box<Stack> {
    Box::new(Stack::Cons(value, stack))
}

fn pop(stack: Box<Stack>) -> (Value, Box<Stack>) {
    match *stack {
        Stack::Cons(ref token, ref rest_of_stack) => {
            (token.clone(), rest_of_stack.clone())
        },
        Stack::Nil => panic!("stack is empty!")
    }
}

fn is_empty(stack: &Stack) -> bool {
    match *stack {
        Stack::Cons(_, _) => false,
        Stack::Nil => true
    }
}

fn add_env(mut env: Box<Env>, key: &str, value: Value) -> Box<Env> {
    env.insert(key.to_string(), value);
    env
}

fn get_env<'a>(env: &'a Env, key: &str) -> &'a Value {
    env.get(&key.to_string()).unwrap()
}

fn eval_if(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (c2, s) = pop(stack);
    let (c1, s) = pop(s);
    let (pred, s) = pop(s);
    if get_boolean(&pred) {
        match c1 {
            Value::ValClosure(e, f) => {
                let (_, s) = do_evaluate(e, s, &f);
                (env, s)
            },
            _ => panic!("can't use non-function {:?} as if function", c1)
        }
    } else {
        match c2 {
            Value::ValClosure(e, f) => {
                let (_, s) = do_evaluate(e, s, &f);
                (env, s)
            },
            _ => panic!("can't use non-function {:?} as if function", c2)
        }
    }
}

fn eval_apply(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (c, s) = pop(stack);
    match c {
        Value::ValClosure(e, f) => {
            let (_, s) = do_evaluate(e, s, &f);
            (env, s)
        },
        _ => panic!("can't apply non-function {:?}", c)
    }
}

fn eval_addi(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (i2, s) = pop(stack);
    let (i1, s) = pop(s);
    (env, push(s, Value::ValInteger(get_integer(&i1) + get_integer(&i2))))
}

fn eval_addf(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (r2, s) = pop(stack);
    let (r1, s) = pop(s);
    (env, push(s, Value::ValReal(get_real(&r1) + get_real(&r2))))
}

fn eval_acos(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (r, s) = pop(stack);
    (env, push(s, Value::ValReal(get_real(&r).acos().to_degrees())))
}

fn eval_asin(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (r, s) = pop(stack);
    (env, push(s, Value::ValReal(get_real(&r).asin().to_degrees())))
}

fn eval_clampf(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (r, s) = pop(stack);
    (env, push(s, Value::ValReal(get_real(&r).max(0.0).min(1.0))))
}

fn eval_cos(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (r, s) = pop(stack);
    let mut res = get_real(&r).to_radians().cos();
    if res.abs() < 1e-15 {
        res = 0.0;
    }
    (env, push(s, Value::ValReal(res)))
}

fn eval_divi(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (i2, s) = pop(stack);
    let (i1, s) = pop(s);
    (env, push(s, Value::ValInteger(get_integer(&i1) / get_integer(&i2))))
}

fn eval_divf(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (r2, s) = pop(stack);
    let (r1, s) = pop(s);
    (env, push(s, Value::ValReal(get_real(&r1) / get_real(&r2))))
}

fn eval_eqi(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (i2, s) = pop(stack);
    let (i1, s) = pop(s);
    let rv = get_integer(&i1) == get_integer(&i2);
    (env, push(s, Value::ValBoolean(rv)))
}

fn eval_eqf(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (r2, s) = pop(stack);
    let (r1, s) = pop(s);
    let rv = get_real(&r1) == get_real(&r2);
    (env, push(s, Value::ValBoolean(rv)))
}

fn eval_floor(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (r, s) = pop(stack);
    (env, push(s, Value::ValInteger(get_real(&r).floor() as i64)))
}

fn eval_frac(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (r, s) = pop(stack);
    (env, push(s, Value::ValReal(get_real(&r).fract())))
}

fn eval_lessi(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (i2, s) = pop(stack);
    let (i1, s) = pop(s);
    if get_integer(&i1) < get_integer(&i2) {
        (env, push(s, Value::ValBoolean(true)))
    } else {
        (env, push(s, Value::ValBoolean(false)))
    }
}

fn eval_lessf(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (r2, s) = pop(stack);
    let (r1, s) = pop(s);
    if get_real(&r1) < get_real(&r2) {
        (env, push(s, Value::ValBoolean(true)))
    } else {
        (env, push(s, Value::ValBoolean(false)))
    }
}

fn eval_modi(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (i2, s) = pop(stack);
    let (i1, s) = pop(s);
    (env, push(s, Value::ValInteger(modi(get_integer(&i1), get_integer(&i2)))))
}

fn eval_muli(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (i2, s) = pop(stack);
    let (i1, s) = pop(s);
    (env, push(s, Value::ValInteger(get_integer(&i1) * get_integer(&i2))))
}

fn eval_mulf(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (r2, s) = pop(stack);
    let (r1, s) = pop(s);
    (env, push(s, Value::ValReal(get_real(&r1) * get_real(&r2))))
}

fn eval_negi(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (i, s) = pop(stack);
    (env, push(s, Value::ValInteger(-get_integer(&i))))
}

fn eval_negf(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (r, s) = pop(stack);
    (env, push(s, Value::ValReal(-get_real(&r))))
}

fn eval_real(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (i, s) = pop(stack);
    (env, push(s, Value::ValReal(get_integer(&i) as f64)))
}

fn eval_sin(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (r, s) = pop(stack);
    let mut res = get_real(&r).to_radians().sin();
    if res.abs() < 1e-15 {
        res = 0.0;
    }
    (env, push(s, Value::ValReal(res)))
}

fn eval_sqrt(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (r, s) = pop(stack);
    (env, push(s, Value::ValReal(get_real(&r).sqrt())))
}

fn eval_subi(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (i2, s) = pop(stack);
    let (i1, s) = pop(s);
    (env, push(s, Value::ValInteger(get_integer(&i1) - get_integer(&i2))))
}

fn eval_subf(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (r2, s) = pop(stack);
    let (r1, s) = pop(s);
    (env, push(s, Value::ValReal(get_real(&r1) - get_real(&r2))))
}

// def eval_point(env, stack):
//     z, stack = pop(stack)
//     y, stack = pop(stack)
//     x, stack = pop(stack)
//     return env, push(stack, make_point(get_real(x), get_real(y), get_real(z)))

// def eval_getx(env, stack):
//     p, stack = pop(stack)
//     return env, push(stack, make_real(get_point_x(p)))

// def eval_gety(env, stack):
//     p, stack = pop(stack)
//     return env, push(stack, make_real(get_point_y(p)))

// def eval_getz(env, stack):
//     p, stack = pop(stack)
//     return env, push(stack, make_real(get_point_z(p)))

fn get_array<'a>(v: &'a Value) -> &'a Vec<Value> {
    match v {
        &Value::ValArray(ref a) => a,
        other => panic!("{:?} is not an array", other)
    }
}

fn eval_get(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (i, s) = pop(stack);
    let (a, s) = pop(s);
    let iv = get_integer(&i);
    let av = get_array(&a);
    if iv < 0 || iv > av.len() as i64 {
        panic!("array subscript error: len: {:?}, index: {:?}", av.len(), iv);
    }
    (env, push(s, av[iv as usize].clone()))
}

fn eval_length(env: Box<Env>, stack: Box<Stack>) -> (Box<Env>, Box<Stack>) {
    let (a, s) = pop(stack);
    (env, push(s, Value::ValInteger(get_array(&a).len() as i64)))
}

// def eval_sphere(env, stack):
//     surface, stack = pop(stack)
//     return env, push(stack, primitives.Sphere(get_surface(surface)))

// def eval_cube(env, stack):
//     surface, stack = pop(stack)
//     return env, push(stack, primitives.Cube(get_surface(surface)))

// def eval_cylinder(env, stack):
//     surface, stack = pop(stack)
//     return env, push(stack, primitives.Cylinder(get_surface(surface)))

// def eval_cone(env, stack):
//     surface, stack = pop(stack)
//     return env, push(stack, primitives.Cone(get_surface(surface)))

// def eval_plane(env, stack):
//     surface, stack = pop(stack)
//     return env, push(stack, primitives.Plane(get_surface(surface)))

// def eval_union(env, stack):
//     obj2, stack = pop(stack)
//     obj1, stack = pop(stack)
//     return env, push(stack, primitives.Union(obj1, obj2))

// def eval_intersect(env, stack):
//     obj2, stack = pop(stack)
//     obj1, stack = pop(stack)
//     return env, push(stack, primitives.Intersect(obj1, obj2))

// def eval_difference(env, stack):
//     obj2, stack = pop(stack)
//     obj1, stack = pop(stack)
//     return env, push(stack, primitives.Difference(obj1, obj2))

// def eval_translate(env, stack):
//     tz, stack = pop(stack)
//     ty, stack = pop(stack)
//     tx, stack = pop(stack)
//     obj, stack = pop(stack)
//     obj.translate(get_real(tx), get_real(ty), get_real(tz))
//     return env, push(stack, obj)

// def eval_scale(env, stack):
//     sz, stack = pop(stack)
//     sy, stack = pop(stack)
//     sx, stack = pop(stack)
//     obj, stack = pop(stack)
//     obj.scale(get_real(sx), get_real(sy), get_real(sz))
//     return env, push(stack, obj)

// def eval_uscale(env, stack):
//     s, stack = pop(stack)
//     obj, stack = pop(stack)
//     obj.uscale(get_real(s))
//     return env, push(stack, obj)

// def eval_rotatex(env, stack):
//     d, stack = pop(stack)
//     obj, stack = pop(stack)
//     obj.rotatex(get_real(d))
//     return env, push(stack, obj)

// def eval_rotatey(env, stack):
//     d, stack = pop(stack)
//     obj, stack = pop(stack)
//     obj.rotatey(get_real(d))
//     return env, push(stack, obj)

// def eval_rotatez(env, stack):
//     d, stack = pop(stack)
//     obj, stack = pop(stack)
//     obj.rotatez(get_real(d))
//     return env, push(stack, obj)

// def eval_light(env, stack):
//     color, stack = pop(stack)
//     d, stack = pop(stack)
//     return env, push(stack, lights.Light(get_point(d),
//                                          get_point(color)))

// def eval_pointlight(env, stack):
//     color, stack = pop(stack)
//     pos, stack = pop(stack)
//     return env, push(stack, lights.PointLight(get_point(pos),
//                                               get_point(color)))

// def eval_spotlight(env, stack):
//     exp, stack = pop(stack)
//     cutoff, stack = pop(stack)
//     color, stack = pop(stack)
//     at, stack = pop(stack)
//     pos, stack = pop(stack)
//     return env, push(stack, lights.SpotLight(get_point(pos),
//                                              get_point(at),
//                                              get_point(color),
//                                              get_real(cutoff),
//                                              get_real(exp)))

// def eval_render(env, stack):
//     file, stack = pop(stack)
//     ht, stack = pop(stack)
//     wid, stack = pop(stack)
//     fov, stack = pop(stack)
//     depth, stack = pop(stack)
//     obj, stack = pop(stack)
//     lights, stack = pop(stack)
//     amb, stack = pop(stack)
//     raytracer.render(get_point(amb),
//                      get_array(lights),
//                      get_node(obj),
//                      get_integer(depth),
//                      get_real(fov),
//                      get_integer(wid),
//                      get_integer(ht),
//                      get_string(file))
//     return env, stack

// def get_surface(surface):
//     assert check_closure(surface)
//     def do_surface(face, u, v):
//         stack = make_stack()
//         stack = push(stack, make_integer(face))
//         stack = push(stack, make_real(u))
//         stack = push(stack, make_real(v))
//         e, stack, a = do_evaluate(get_closure_env(surface),
//                                   stack,
//                                   get_closure_function(surface))
//         n, stack = pop(stack)
//         ks, stack = pop(stack)
//         kd, stack = pop(stack)
//         sc, stack = pop(stack)
//         return get_point(sc), get_real(kd), get_real(ks), get_real(n)
//     return do_surface

fn do_evaluate(mut env: Box<Env>, mut stack: Box<Stack>, ast: &[parser::AstNode]) -> (Box<Env>, Box<Stack>) {
    for i in 0..ast.len() {
        match &ast[i] {
            &parser::AstNode::Function(ref v) => {
                stack = push(stack, Value::ValClosure(env.clone(), v.clone()));
            },
            &parser::AstNode::Array(ref v) => {
                let (_, s) = do_evaluate(env.clone(), Box::new(Stack::Nil), v);
                stack = push(stack, make_array(s));
            },
            &parser::AstNode::Leaf(ref t) => {
                match t {
                    &tokenizer::Token::Integer(v) => {
                        stack = push(stack, Value::ValInteger(v))
                    },
                    &tokenizer::Token::Real(v) => {
                        stack = push(stack, Value::ValReal(v))
                    },
                    &tokenizer::Token::Boolean(v) => {
                        stack = push(stack, Value::ValBoolean(v))
                    },
                    &tokenizer::Token::Str(ref v) => {
                        stack = push(stack, Value::ValString(v.clone()))
                    },
                    &tokenizer::Token::Binder(ref v) => {
                        let (i, s) = pop(stack);
                        stack = s;
                        env = add_env(env, &v, i);
                    },
                    &tokenizer::Token::Identifier(ref v) => {
                        let val = get_env(&*env, &v);
                        //                 if isinstance(e, primitives.Node):
                        //                     e = copy.deepcopy(e)
                        stack = push(stack, val.clone())
                    },
                    &tokenizer::Token::Operator(ref v) => {
                        let (e, s) = eval_op(v, env, stack);
                        env = e;
                        stack = s;
                    },
                    token => {
                        panic!("evaluate error, unknown token: {:?}", token);
                    }
                }
            }
        }
    }
    (env, stack)
}

fn evaluate(ast: &[parser::AstNode]) -> (Box<Env>, Box<Stack>) {
    // Apparently can't call static methods on aliased types, so her goes the full name of Env
    let env: Box<Env> = Box::new(collections::HashMap::<String, Value>::new());
    do_evaluate(env, Box::new(Stack::Nil), &ast)
}

pub fn run(gml: &str) -> (Box<Env>, Box<Stack>) {
    evaluate(&parser::parse(&tokenizer::tokenize(gml)))
}

#[test]
fn test_evaluator() {
    let  (env, stack) = run("1 /x");
    println!("env: {:?}, stack: {:?}", *env, *stack);
    assert_eq!(env.get("x").unwrap(), &Value::ValInteger(1));
    let  (env, stack) = run(r#""apa" /x"#);
    println!("env: {:?}, stack: {:?}", env, stack);
    assert_eq!(env.get("x").unwrap(), &Value::ValString("apa".to_string()));
    assert_eq!(*get_string(env.get("x").unwrap()), "apa".to_string());
    let  (env, stack) = run("1 2 addi");
    println!("env: {:?}, stack: {:?}", env, stack);
    assert_eq!(stack, Box::new(Stack::Cons(Value::ValInteger(3), Box::new(Stack::Nil))));
    let  (env, stack) = run("1.5 2.5 addf");
    println!("env: {:?}, stack: {:?}", env, stack);
    assert_eq!(stack, Box::new(Stack::Cons(Value::ValReal(4.0), Box::new(Stack::Nil))));
    let  (env, stack) = run("1 /x x x addi");
    println!("env: {:?}, stack: {:?}", env, stack);
    assert_eq!(stack, Box::new(Stack::Cons(Value::ValInteger(2), Box::new(Stack::Nil))));
    let  (env, stack) = run("1 { /x x x } apply");
    println!("env: {:?}, stack: {:?}", env, stack);
    assert_eq!(stack, Box::new(Stack::Cons(Value::ValInteger(1), Box::new(Stack::Cons(Value::ValInteger(1), Box::new(Stack::Nil))))));
    let  (env, stack) = run("true { 1 } { 2 } if");
    println!("env: {:?}, stack: {:?}", env, stack);
    assert_eq!(stack, Box::new(Stack::Cons(Value::ValInteger(1), Box::new(Stack::Nil))));
    let  (env, stack) = run("false { 1 } { 2 } if");
    println!("env: {:?}, stack: {:?}", env, stack);
    assert_eq!(stack, Box::new(Stack::Cons(Value::ValInteger(2), Box::new(Stack::Nil))));
    let  (env, stack) = run("1 2 eqi");
    println!("env: {:?}, stack: {:?}", env, stack);
    assert_eq!(stack, Box::new(Stack::Cons(Value::ValBoolean(false), Box::new(Stack::Nil))));
    let  (env, stack) = run("5 5 eqi");
    println!("env: {:?}, stack: {:?}", env, stack);
    assert_eq!(stack, Box::new(Stack::Cons(Value::ValBoolean(true), Box::new(Stack::Nil))));
    let  (env, stack) = run("1.5 2.7 eqf");
    println!("env: {:?}, stack: {:?}", env, stack);
    assert_eq!(stack, Box::new(Stack::Cons(Value::ValBoolean(false), Box::new(Stack::Nil))));
    let  (env, stack) = run("5.123 5.123 eqf");
    println!("env: {:?}, stack: {:?}", env, stack);
    assert_eq!(stack, Box::new(Stack::Cons(Value::ValBoolean(true), Box::new(Stack::Nil))));
    let  (env, stack) = run("2 1 lessi");
    println!("env: {:?}, stack: {:?}", env, stack);
    assert_eq!(stack, Box::new(Stack::Cons(Value::ValBoolean(false), Box::new(Stack::Nil))));
    let  (env, stack) = run("2 2 lessi");
    println!("env: {:?}, stack: {:?}", env, stack);
    assert_eq!(stack, Box::new(Stack::Cons(Value::ValBoolean(false), Box::new(Stack::Nil))));
    let  (env, stack) = run("1 2 lessi");
    println!("env: {:?}, stack: {:?}", env, stack);
    assert_eq!(stack, Box::new(Stack::Cons(Value::ValBoolean(true), Box::new(Stack::Nil))));
    let  (env, stack) = run("2.0 1.0 lessf");
    println!("env: {:?}, stack: {:?}", env, stack);
    assert_eq!(stack, Box::new(Stack::Cons(Value::ValBoolean(false), Box::new(Stack::Nil))));
    let  (env, stack) = run("2.0 2.0 lessf");
    println!("env: {:?}, stack: {:?}", env, stack);
    assert_eq!(stack, Box::new(Stack::Cons(Value::ValBoolean(false), Box::new(Stack::Nil))));
    let  (env, stack) = run("1.0 2.0 lessf");
    println!("env: {:?}, stack: {:?}", env, stack);
    assert_eq!(stack, Box::new(Stack::Cons(Value::ValBoolean(true), Box::new(Stack::Nil))));
    let  (env, stack) = run("false /b b { 1 } { 2 } if");
    println!("env: {:?}, stack: {:?}", env, stack);
    let  (env, stack) = run("4 /x 2 x addi");
    println!("env: {:?}, stack: {:?}", env, stack);
    let  (env, stack) = run("1 { /x x x } apply addi");
    println!("env: {:?}, stack: {:?}", env, stack);
    let  (env, stack) = run("{ /x x x } /dup { dup apply muli } /sq 3 sq apply");
    println!("env: {:?}, stack: {:?}", env, stack);
    let  (env, stack) = run("{ /x /y x y } /swap 3 4 swap apply");
    println!("env: {:?}, stack: {:?}", env, stack);
    let  (env, stack) = run("{ /self /n n 2 lessi { 1 } { n 1 subi self self apply n muli } if } /fact 12 fact fact apply");
    println!("env: {:?}, stack: {:?}", env, stack);
    for ui in 0..10 {
        for vi in 0..10 {
            let u = ui as f64;
            let v = vi as f64;
            // Implement a small program
            // Escaping { in format strings is done by adding an extra brace -> {{
            let prog = format!("1 /col1 2 /col2 {:.1} /u {:.1} /v {{ /y /x x x mulf y y mulf addf sqrt }} /dist \n \
            {{ \n \
            u 0.5 subf /u v 0.5 subf /v \n \
            u u v dist apply divf /b \n \
            0.0 v lessf {{ b asin }} {{ 360.0 b asin subf }} if 180.0 addf 30.0 divf \n \
            floor 2 modi 1 eqi {{ col1 }} {{ col2 }} if \n \
            }} apply", u / 10.0, v / 10.0);
            // Implement the same program in rust:
            fn test(u: f64, v: f64) -> i64 {
                let u = u - 0.5;
                let v = v - 0.5;
                let b = u / (u * u + v * v).sqrt();
                let mut c: f64;
                if 0.0 < v {
                    c = b.asin().to_degrees();
                } else {
                    c = 360.0 - b.asin().to_degrees();
                }
                c = c + 180.0;
                c = c / 30.0;
                let mut ci = c.floor() as i64;
                ci = modi(ci, 2);
                if ci == 1 {
                    return 1;
                } else {
                    return 2;
                }
            }
            // Compare results (which should be the only thing om the top of the stack)
            let expected = test(u / 10.0, v / 10.0);
            let (_, stack) = run(prog.as_ref());
            match *stack {
                Stack::Cons(ref val, ref rest) => {
                    assert_eq!(get_integer(&val), expected);
                    assert_eq!(**rest, Stack::Nil);
                },
                _ => panic!()
            }
        }
    }
    let  (env, _) = run("-0.4 clampf /x");
    assert_eq!(env.get("x").unwrap(), &Value::ValReal(0.0));
    let  (env, _) = run("1.1 clampf /x");
    assert_eq!(env.get("x").unwrap(), &Value::ValReal(1.0));
    let  (env, _) = run("0.8 clampf /x");
    assert_eq!(env.get("x").unwrap(), &Value::ValReal(0.8));
    let  (env, _) = run("[1 2 3] length /x");
    assert_eq!(env.get("x").unwrap(), &Value::ValInteger(3));
    let  (env, _) = run("[1 2 3] 1 get /x");
    assert_eq!(env.get("x").unwrap(), &Value::ValInteger(2));
}
