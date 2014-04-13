extern crate collections;

pub mod parser;

#[deriving(Eq, Show, Clone)]
enum Value {
    ValClosure(~Env, ~[parser::AstNode]), // The ast for the function
    ValArray, // TODO
    ValBoolean(bool),
    ValReal(f64),
    ValInteger(i64),
    ValString(~str)
}

type Env = collections::hashmap::HashMap<~str, Value>;

#[deriving(Eq, Show, Clone)]
enum Stack {
    Cons(Value, ~Stack),
    Nil
}

// def divi(a, b):
//     rv = a // b
//     if rv < 0:
//         rv += 1 # we need to round towards zero, which python doesn't
//     return rv

// def modi(a, b):
//     return a - b * divi(a, b)

fn get_integer(v: &Value) -> i64 {
    match v {
        &ValInteger(i) => i,
        other => fail!("{} is not an integer", other)
    }
}

fn get_string(v: &Value) -> ~str {
    match v {
        &ValString(ref i) => i.clone(),
        other => fail!("{} is not an integer", other)
    }
}

fn get_real(v: &Value) -> f64 {
    match v {
        &ValReal(f) => f,
        other => fail!("{} is not an real", other)
    }
}

fn get_boolean(v: &Value) -> bool {
    match v {
        &ValBoolean(b) => b,
        other => fail!("{} is not an boolean", other)
    }
}

// def make_array(v):
//     tmp = []
//     while v:
//         e, v = pop(v)
//         tmp.insert(0, e)
//     return ('Array', tmp)

// def get_array(a):
//     assert check_array(a)
//     return get_value(a)

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

fn push(stack: ~Stack, value: Value) -> ~Stack {
    ~Cons(value, stack)
}

fn pop(stack: ~Stack) -> (Value, ~Stack) {
    println!("pop from stack: {}", stack);
    match stack {
        ~Cons(token, rest_of_stack) => {
            (token, rest_of_stack.clone())
        },
        ~Nil => fail!("stack is empty!")
    }
}

fn add_env(mut env: ~Env, key: ~str, value: Value) -> ~Env {
    env.insert(key, value);
    env
}

fn get_env<'a>(env: &'a Env, key: &'a ~str) -> &'a Value {
    env.get(key)
}

fn eval_if(env: ~Env, stack: ~Stack) -> (~Env, ~Stack) {
    let (c2, s) = pop(stack);
    let (c1, s) = pop(s);
    let (pred, s) = pop(s);
    if get_boolean(&pred) {
        match c1 {
            ValClosure(e, f) => {
                let (e, s) = do_evaluate(e, s, f);
                (env, s)
            },
            _ => fail!("can't use non-function {} as if function", c1)
        }
    } else {
        match c2 {
            ValClosure(e, f) => {
                let (e, s) = do_evaluate(e, s, f);
                (env, s)
            },
            _ => fail!("can't use non-function {} as if function", c2)
        }
    }
}

fn eval_apply(env: ~Env, stack: ~Stack) -> (~Env, ~Stack) {
    let (c, s) = pop(stack);
    match c {
        ValClosure(e, f) => {
            let (e, s) = do_evaluate(e, s, f);
            (env, s)
        },
        _ => fail!("can't apply non-function {}", c)
    }
}

fn eval_addi(env: ~Env, stack: ~Stack) -> (~Env, ~Stack) {
    let (i2, s) = pop(stack);
    let (i1, s) = pop(s);
    (env, push(s, ValInteger(get_integer(&i1) + get_integer(&i2))))
}

fn eval_addf(env: ~Env, stack: ~Stack) -> (~Env, ~Stack) {
    let (r2, s) = pop(stack);
    let (r1, s) = pop(s);
    (env, push(s, ValReal(get_real(&r1) + get_real(&r2))))
}

fn eval(op: &parser::tokenizer::Ops, env: ~Env, stack: ~Stack) -> (~Env, ~Stack) {
    match op {
        &parser::tokenizer::OpAddi => eval_addi(env, stack),
        &parser::tokenizer::OpAddf => eval_addf(env, stack),
        &parser::tokenizer::OpApply => eval_apply(env, stack),
        &parser::tokenizer::OpIf => eval_if(env, stack),
        _ => fail!("operator {} not implemented yet!", op)
    }
}

// def eval_acos(env, stack):
//     r, stack = pop(stack)
//     return env, push(stack, make_real(math.degrees(math.acos(get_real(r)))))
    
// def eval_asin(env, stack):
//     r, stack = pop(stack)
//     return env, push(stack, make_real(math.degrees(math.asin(get_real(r)))))

// def eval_clampf(env, stack):
//     r, stack = pop(stack)
//     rv = get_real(r)
//     if rv < 0.0:
//         rv = 0.0
//     elif rv > 1.0:
//         rv = 1.0
//     return env, push(stack, make_real(rv))

// def eval_cos(env, stack):
//     r, stack = pop(stack)
//     res = math.cos(math.radians(get_real(r)))
//     if abs(res) < 1e-15:
//         res = 0.0
//     return env, push(stack, make_real(res))

// def eval_divi(env, stack):
//     i2, stack = pop(stack)
//     i1, stack = pop(stack)
//     return env, push(stack, make_integer(divi(get_integer(i1), get_integer(i2))))

// def eval_divf(env, stack):
//     r2, stack = pop(stack)
//     r1, stack = pop(stack)
//     rv = get_real(r1) / get_real(r2)
//     return env, push(stack, make_real(rv))

// def eval_eqi(env, stack):
//     i2, stack = pop(stack)
//     i1, stack = pop(stack)
//     rv = False
//     if get_integer(i1) == get_integer(i2):
//         rv = True        
//     return env, push(stack, make_boolean(rv))

// def eval_eqf(env, stack):
//     r2, stack = pop(stack)
//     r1, stack = pop(stack)
//     rv = False
//     if get_real(r1) == get_real(r2):
//         rv = True        
//     return env, push(stack, make_boolean(rv))

// def eval_floor(env, stack):
//     r, stack = pop(stack)
//     return env, push(stack, make_integer(int(math.floor(get_real(r)))))

// def eval_frac(env, stack):
//     r, stack = pop(stack)
//     return env, push(stack, make_real(math.modf(get_real(r))[0]))

// def eval_lessi(env, stack):
//     i2, stack = pop(stack)
//     i1, stack = pop(stack)
//     rv = False
//     if get_integer(i1) < get_integer(i2):
//         rv = True
//     return env, push(stack, make_boolean(rv))

// def eval_lessf(env, stack):
//     r2, stack = pop(stack)
//     r1, stack = pop(stack)
//     rv = False
//     if get_real(r1) < get_real(r2):
//         rv = True
//     return env, push(stack, make_boolean(rv))

// def eval_modi(env, stack):
//     i2, stack = pop(stack)
//     i1, stack = pop(stack)
//     return env, push(stack, make_integer(modi(get_integer(i1), get_integer(i2))))

// def eval_muli(env, stack):
//     i2, stack = pop(stack)
//     i1, stack = pop(stack)
//     return env, push(stack, make_integer(get_integer(i1)*get_integer(i2)))

// def eval_mulf(env, stack):
//     r2, stack = pop(stack)
//     r1, stack = pop(stack)
//     return env, push(stack, make_real(get_real(r1)*get_real(r2)))

// def eval_negi(env, stack):
//     i, stack = pop(stack)
//     return env, push(stack, make_integer(-get_integer(i)))

// def eval_negf(env, stack):
//     r, stack = pop(stack)
//     return env, push(stack, make_real(-get_real(r)))

// def eval_real(env, stack):
//     i, stack = pop(stack)
//     return env, push(stack, make_real(float(get_integer(i))))

// def eval_sin(env, stack):
//     r, stack = pop(stack)
//     res = math.sin(math.radians(get_real(r)))
//     if abs(res) < 1e-15:
//         res = 0.0
//     return env, push(stack, make_real(res))

// def eval_sqrt(env, stack):
//     r, stack = pop(stack)
//     return env, push(stack, make_real(math.sqrt(get_real(r))))

// def eval_subi(env, stack):
//     i2, stack = pop(stack)
//     i1, stack = pop(stack)
//     return env, push(stack, make_integer(get_integer(i1)-get_integer(i2)))    

// def eval_subf(env, stack):
//     r2, stack = pop(stack)
//     r1, stack = pop(stack)
//     return env, push(stack, make_real(get_real(r1)-get_real(r2)))

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

// def eval_get(env, stack):
//     i, stack = pop(stack)
//     a, stack = pop(stack)
//     iv = get_integer(i)
//     av = get_array(a)
//     if iv < 0 or iv > len(av):
//         raise GMLSubscriptError
//     try:
//         return env, push(stack, av[iv])
//     except:
//         print iv
//         raise

// def eval_length(env, stack):
//     a, stack = pop(stack)
//     return env, push(stack, make_integer(len(get_array(a))))

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

fn do_evaluate(mut env: ~Env, mut stack: ~Stack, ast: &[parser::AstNode]) -> (~Env, ~Stack) {
    for i in range(0, ast.len()) {
        match &ast[i] {
            &parser::Function(ref v) => {
                stack = push(stack, ValClosure(env.clone(), v.clone()));
            },
            &parser::Array(_) => {
//             e, s, a = do_evaluate(env, make_stack(), v)
//             stack = push(stack, make_array(s))
            },
            &parser::Leaf(ref t) => {
                match t {
                    &parser::tokenizer::Integer(v) => {
                        stack = push(stack, ValInteger(v))
                    },
                    &parser::tokenizer::Real(v) => {
                        stack = push(stack, ValReal(v))
                    },
                    &parser::tokenizer::Boolean(v) => {
                        stack = push(stack, ValBoolean(v))
                    },
                    &parser::tokenizer::String(ref v) => {
                        stack = push(stack, ValString(v.clone()))
                    },
                    &parser::tokenizer::Binder(ref v) => {
                        let (i, s) = pop(stack);
                        stack = s;
                        env = add_env(env, v.clone(), i);
                    },
                    &parser::tokenizer::Identifier(ref v) => {
                        let val = get_env(env, v);
                        //                 if isinstance(e, primitives.Node):
                        //                     e = copy.deepcopy(e)
                        stack = push(stack, val.clone())
                    },
                    &parser::tokenizer::Operator(ref v) => {
                        let (e, s) = eval(v, env, stack);
                        env = e;
                        stack = s;
                    },
                    token => {
                        fail!("evaluate error, unknown token: {}", token);
                    }
                }
            }
        }
    }
    (env, stack)
}

// def test(ast, res):    
//     try:
//         t = evaluate(ast)    
//         if t == res:
//             pass
//         else:
//             print ast,"!=",res
//             print ast,"==",t
//     except Exception, e:
//         if type(res) != type or not isinstance(e, res):
//             raise

fn evaluate(ast: ~[parser::AstNode]) -> (~Env, ~Stack) {
    // Apparently can't call static methods on aliased types, so her goes the full name of Env
    let env: ~Env = ~collections::hashmap::HashMap::<~str, Value>::new();
    do_evaluate(env, ~Nil, ast)
}

pub fn run(gml: &str) -> (~Env, ~Stack) {
    evaluate(parser::parse(parser::tokenizer::tokenize(gml)))
}

#[test]
fn test_evaluator() {
    let  (env, stack) = run("1 /x");
    println!("env: {}, stack: {}", env, stack);
    assert_eq!(env.get(&~"x"), &ValInteger(1));
    let  (env, stack) = run("1 2 addi");
    println!("env: {}, stack: {}", env, stack);
    assert_eq!(stack, ~Cons(ValInteger(3), ~Nil));
    let  (env, stack) = run("1.5 2.5 addf");
    println!("env: {}, stack: {}", env, stack);
    assert_eq!(stack, ~Cons(ValReal(4.0), ~Nil));
    let  (env, stack) = run("1 /x x x addi");
    println!("env: {}, stack: {}", env, stack);
    assert_eq!(stack, ~Cons(ValInteger(2), ~Nil));
    let  (env, stack) = run("1 { /x x x } apply");
    println!("env: {}, stack: {}", env, stack);
    assert_eq!(stack, ~Cons(ValInteger(1), ~Cons(ValInteger(1), ~Nil)));
    let  (env, stack) = run("true { 1 } { 2 } if");
    println!("env: {}, stack: {}", env, stack);
    assert_eq!(stack, ~Cons(ValInteger(1), ~Nil));
    let  (env, stack) = run("false { 1 } { 2 } if");
    println!("env: {}, stack: {}", env, stack);
    assert_eq!(stack, ~Cons(ValInteger(2), ~Nil));
//     env, stack, ast = run("false /b b { 1 } { 2 } if")
//     print stack
//     env, stack, ast = run("4 /x 2 x addi")
//     print stack
//     env, stack, ast = run("1 { /x x x } apply addi")
//     print stack
//     env, stack, ast = run("{ /x x x } /dup { dup apply muli } /sq 3 sq apply")
//     print stack
//     env, stack, ast = run("{ /x /y x y } /swap 3 4 swap apply")
//     print stack
//     env, stack, ast = run("{ /self /n n 2 lessi { 1 } { n 1 subi self self apply n muli } if } /fact 12 fact fact apply")
//     print stack
//     for u in range(10):
//         for v in range(10):
//             prog = """1 /col1 2 /col2 %f /u %f /v { /y /x x x mulf y y mulf addf sqrt } /dist
//             { 
//             u 0.5 subf /u v 0.5 subf /v
//             u u v dist apply divf /b
//             0.0 v lessf { b asin } { 360.0 b asin subf } if 180.0 addf 30.0 divf
//             floor 2 modi 1 eqi { col1 } { col2 } if
//             } apply"""%(u / 10.0, v / 10.0)
//             def test(u, v):
//                 u = u - 0.5
//                 v = v - 0.5
//                 b = u / (math.sqrt(u*u+v*v))
//                 if 0.0 < v:
//                     c = math.degrees(math.asin(b))
//                 else:
//                     c = 360 - math.degrees(math.asin(b))
//                 c = c + 180.8
//                 c = c / 30.0
//                 c = math.floor(c)
//                 c = modi(c, 2)
//                 if c == 1:
//                     print 1,
//                 else:
//                     print 2,
//             #test(u / 10.0, v / 10.0)
//             env, stack, ast = run(prog)
//             print stack,
//         print
}
