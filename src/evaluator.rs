use std::collections;
use std::error::Error as StdError;
use std::fmt;
use std::rc::Rc;
use parser;
use primitives;
use tokenizer;
use vecmath;
use lights;
use raytracer;

#[derive(Clone)]
pub enum Value {
    ValClosure(Env, Vec<parser::AstNode>), // The ast for the function
    ValArray(Vec<Value>),
    ValBoolean(bool),
    ValReal(f64),
    ValInteger(i64),
    ValString(String),
    ValPoint(vecmath::Vec3),
    ValNode(Box<primitives::Node>),
    ValLight(Box<lights::Light>),
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &Value::ValNode(_) => write!(f, "ValNode"),
            &Value::ValLight(_) => write!(f, "ValLight"),
            &Value::ValClosure(_, _) => write!(f, "ValClosure"),
            &Value::ValBoolean(x) => write!(f, "ValBoolean {{ {} }}", x),
            &Value::ValReal(x) => write!(f, "ValReal {{ {} }}", x),
            &Value::ValInteger(x) => write!(f, "ValInteger {{ {} }}", x),
            &Value::ValString(ref x) => write!(f, "ValString {{ {} }}", x),
            &Value::ValArray(ref x) => write!(f, "ValArray {{ {:?} }}", x),
            &Value::ValPoint(ref x) => write!(f, "ValPoint {{ {:?} }}", x),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Value) -> bool {
        match (self, other) {
            (&Value::ValNode(_), &Value::ValNode(_)) => false,
            (&Value::ValLight(_), &Value::ValLight(_)) => false,
            (&Value::ValClosure(ref se, ref sa), &Value::ValClosure(ref oe, ref oa)) => {
                se == oe && sa == oa
            }
            (&Value::ValBoolean(sv), &Value::ValBoolean(ov)) => sv == ov,
            (&Value::ValReal(sv), &Value::ValReal(ov)) => sv == ov,
            (&Value::ValInteger(sv), &Value::ValInteger(ov)) => sv == ov,
            (&Value::ValString(ref sv), &Value::ValString(ref ov)) => sv == ov,
            (&Value::ValArray(ref sv), &Value::ValArray(ref ov)) => sv == ov,
            (&Value::ValPoint(ref sv), &Value::ValPoint(ref ov)) => sv == ov,
            _ => false,
        }
    }
}

pub type Env = collections::HashMap<String, Value>;
pub type Stack = Vec<Value>;

fn pop(stack: &mut Stack) -> Result<Value, EvalError> {
    return stack.pop().ok_or(EvalError::EmptyStack);
}

#[derive(Debug)]
enum EvalError {
    EmptyStack,
    WrongType(Value),
    OpNotImplemented(tokenizer::Ops),
    ArrayOutOfBounds(i64, usize),
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        return f.write_str(self.description());
    }
}

impl StdError for EvalError {
    fn description(&self) -> &str {
        match *self {
            EvalError::EmptyStack => "EmptyStack",
            EvalError::WrongType(_) => "Wrongtype",
            EvalError::OpNotImplemented(_) => "OpNotImplemented",
            EvalError::ArrayOutOfBounds(_, _) => "ArrayOutOfBounds",
        }
    }
}

fn eval_op(op: &tokenizer::Ops, stack: &mut Stack) -> Result<(), EvalError> {
    match op {
        &tokenizer::Ops::OpAddi => eval_addi(stack),
        &tokenizer::Ops::OpAddf => eval_addf(stack),
        &tokenizer::Ops::OpApply => eval_apply(stack),
        &tokenizer::Ops::OpAcos => eval_acos(stack),
        &tokenizer::Ops::OpAsin => eval_asin(stack),
        &tokenizer::Ops::OpClampf => eval_clampf(stack),
        &tokenizer::Ops::OpCos => eval_cos(stack),
        &tokenizer::Ops::OpDivf => eval_divf(stack),
        &tokenizer::Ops::OpDivi => eval_divi(stack),
        &tokenizer::Ops::OpEqi => eval_eqi(stack),
        &tokenizer::Ops::OpEqf => eval_eqf(stack),
        &tokenizer::Ops::OpFloor => eval_floor(stack),
        &tokenizer::Ops::OpFrac => eval_frac(stack),
        &tokenizer::Ops::OpGet => eval_get(stack),
        &tokenizer::Ops::OpGetx => eval_getx(stack),
        &tokenizer::Ops::OpGety => eval_gety(stack),
        &tokenizer::Ops::OpGetz => eval_getz(stack),
        &tokenizer::Ops::OpIf => eval_if(stack),
        &tokenizer::Ops::OpLength => eval_length(stack),
        &tokenizer::Ops::OpLessi => eval_lessi(stack),
        &tokenizer::Ops::OpLessf => eval_lessf(stack),
        &tokenizer::Ops::OpLight => eval_light(stack),
        &tokenizer::Ops::OpModi => eval_modi(stack),
        &tokenizer::Ops::OpMuli => eval_muli(stack),
        &tokenizer::Ops::OpMulf => eval_mulf(stack),
        &tokenizer::Ops::OpNegi => eval_negi(stack),
        &tokenizer::Ops::OpPoint => eval_point(stack),
        &tokenizer::Ops::OpNegf => eval_negf(stack),
        &tokenizer::Ops::OpReal => eval_real(stack),
        &tokenizer::Ops::OpRender => eval_render(stack),
        &tokenizer::Ops::OpSin => eval_sin(stack),
        &tokenizer::Ops::OpSphere => eval_sphere(stack),
        &tokenizer::Ops::OpSqrt => eval_sqrt(stack),
        &tokenizer::Ops::OpSubi => eval_subi(stack),
        &tokenizer::Ops::OpSubf => eval_subf(stack),
        &tokenizer::Ops::OpTranslate => eval_translate(stack),
        &tokenizer::Ops::OpScale => eval_scale(stack),
        &tokenizer::Ops::OpUscale => eval_uscale(stack),
        &tokenizer::Ops::OpRotatex => eval_rotatex(stack),
        &tokenizer::Ops::OpRotatey => eval_rotatey(stack),
        &tokenizer::Ops::OpRotatez => eval_rotatez(stack),
        &tokenizer::Ops::OpUnion => eval_union(stack),
        &tokenizer::Ops::OpIntersect => eval_intersect(stack),
        &tokenizer::Ops::OpDifference => eval_difference(stack),
        &tokenizer::Ops::OpPlane => eval_plane(stack),
        &tokenizer::Ops::OpPointlight => eval_pointlight(stack),
        op => Err(EvalError::OpNotImplemented(op.clone())),
    }
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
        other => panic!("{:?} is not an integer", other),
    }
}

fn get_string<'a>(v: &'a Value) -> &'a String {
    match v {
        &Value::ValString(ref i) => &i,
        other => panic!("{:?} is not an integer", other),
    }
}

fn get_real(v: &Value) -> f64 {
    match v {
        &Value::ValReal(f) => f,
        other => panic!("{:?} is not an real", other),
    }
}

fn get_boolean(v: &Value) -> bool {
    match v {
        &Value::ValBoolean(b) => b,
        other => panic!("{:?} is not an boolean", other),
    }
}

fn make_array(s: &Stack) -> Value {
    let mut tmp = s.clone();
    tmp.reverse();
    Value::ValArray(tmp)
}

fn get_point<'a>(v: &'a Value) -> &'a vecmath::Vec3 {
    match v {
        &Value::ValPoint(ref p) => &p,
        other => panic!("{:?} is not a point", other),
    }
}

fn move_point(v: Value) -> vecmath::Vec3 {
    match v {
        Value::ValPoint(p) => p,
        other => panic!("{:?} is not a point", other),
    }
}

fn get_point_x(v: &Value) -> f64 {
    get_point(v)[0]
}

fn get_point_y(v: &Value) -> f64 {
    get_point(v)[1]
}

fn get_point_z(v: &Value) -> f64 {
    get_point(v)[2]
}

fn move_node(v: Value) -> Box<primitives::Node> {
    match v {
        Value::ValNode(n) => n,
        _ => panic!("not a node!")
    }
}

fn eval_if(stack: &mut Stack) -> Result<(), EvalError> {
    let c2 = pop(stack)?;
    let c1 = pop(stack)?;
    let pred = pop(stack)?;
    if get_boolean(&pred) {
        match c1 {
            Value::ValClosure(mut e, f) => {
                Ok(do_evaluate(&mut e, stack, &f))
            }
            other => Err(EvalError::WrongType(other)),
        }
    } else {
        match c2 {
            Value::ValClosure(mut e, f) => {
                Ok(do_evaluate(&mut e, stack, &f))
            }
            other => Err(EvalError::WrongType(other)),
        }
    }
}

fn eval_apply(stack: &mut Stack) -> Result<(), EvalError> {
    let c = pop(stack)?;
    match c {
        Value::ValClosure(mut e, f) => {
            Ok(do_evaluate(&mut e, stack, &f))
        }
        other => Err(EvalError::WrongType(other)),
    }
}

fn eval_addi(stack: &mut Stack) -> Result<(), EvalError> {
    let i2 = pop(stack)?;
    let i1 = pop(stack)?;
    stack.push(Value::ValInteger(get_integer(&i1) + get_integer(&i2)));
    Ok(())
}

fn eval_addf(stack: &mut Stack) -> Result<(), EvalError> {
    let r2 = pop(stack)?;
    let r1 = pop(stack)?;
    stack.push(Value::ValReal(get_real(&r1) + get_real(&r2)));
    Ok(())
}

fn eval_acos(stack: &mut Stack) -> Result<(), EvalError> {
    let r = pop(stack)?;
    stack.push(Value::ValReal(get_real(&r).acos().to_degrees()));
    Ok(())
}

fn eval_asin(stack: &mut Stack) -> Result<(), EvalError> {
    let r = pop(stack)?;
    stack.push(Value::ValReal(get_real(&r).asin().to_degrees()));
    Ok(())
}

fn eval_clampf(stack: &mut Stack) -> Result<(), EvalError> {
    let r = pop(stack)?;
    stack.push(Value::ValReal(get_real(&r).max(0.0).min(1.0)));
    Ok(())
}

fn eval_cos(stack: &mut Stack) -> Result<(), EvalError> {
    let r = pop(stack)?;
    let mut res = get_real(&r).to_radians().cos();
    if res.abs() < 1e-15 {
        res = 0.0;
    }
    stack.push(Value::ValReal(res));
    Ok(())
}

fn eval_divi(stack: &mut Stack) -> Result<(), EvalError> {
    let i2 = pop(stack)?;
    let i1 = pop(stack)?;
    stack.push(Value::ValInteger(get_integer(&i1) / get_integer(&i2)));
    Ok(())
}

fn eval_divf(stack: &mut Stack) -> Result<(), EvalError> {
    let r2 = pop(stack)?;
    let r1 = pop(stack)?;
    stack.push(Value::ValReal(get_real(&r1) / get_real(&r2)));
    Ok(())
}

fn eval_eqi(stack: &mut Stack) -> Result<(), EvalError> {
    let i2 = pop(stack)?;
    let i1 = pop(stack)?;
    let rv = get_integer(&i1) == get_integer(&i2);
    stack.push(Value::ValBoolean(rv));
    Ok(())
}

fn eval_eqf(stack: &mut Stack) -> Result<(), EvalError> {
    let r2 = pop(stack)?;
    let r1 = pop(stack)?;
    let rv = get_real(&r1) == get_real(&r2);
    stack.push(Value::ValBoolean(rv));
    Ok(())
}

fn eval_floor(stack: &mut Stack) -> Result<(), EvalError> {
    let r = pop(stack)?;
    stack.push(Value::ValInteger(get_real(&r).floor() as i64));
    Ok(())
}

fn eval_frac(stack: &mut Stack) -> Result<(), EvalError> {
    let r = pop(stack)?;
    stack.push(Value::ValReal(get_real(&r).fract()));
    Ok(())
}

fn eval_lessi(stack: &mut Stack) -> Result<(), EvalError> {
    let i2 = pop(stack)?;
    let i1 = pop(stack)?;
    if get_integer(&i1) < get_integer(&i2) {
        stack.push(Value::ValBoolean(true));
    } else {
        stack.push(Value::ValBoolean(false));
    }
    Ok(())
}

fn eval_lessf(stack: &mut Stack) -> Result<(), EvalError> {
    let r2 = pop(stack)?;
    let r1 = pop(stack)?;
    if get_real(&r1) < get_real(&r2) {
        stack.push(Value::ValBoolean(true));
    } else {
        stack.push(Value::ValBoolean(false));
    }
    Ok(())
}

fn eval_modi(stack: &mut Stack) -> Result<(), EvalError> {
    let i2 = pop(stack)?;
    let i1 = pop(stack)?;
    stack.push(Value::ValInteger(modi(get_integer(&i1), get_integer(&i2))));
    Ok(())
}

fn eval_muli(stack: &mut Stack) -> Result<(), EvalError> {
    let i2 = pop(stack)?;
    let i1 = pop(stack)?;
    stack.push(Value::ValInteger(get_integer(&i1) * get_integer(&i2)));
    Ok(())
}

fn eval_mulf(stack: &mut Stack) -> Result<(), EvalError> {
    let r2 = pop(stack)?;
    let r1 = pop(stack)?;
    stack.push(Value::ValReal(get_real(&r1) * get_real(&r2)));
    Ok(())
}

fn eval_negi(stack: &mut Stack) -> Result<(), EvalError> {
    let i = pop(stack)?;
    stack.push(Value::ValInteger(-get_integer(&i)));
    Ok(())
}

fn eval_negf(stack: &mut Stack) -> Result<(), EvalError> {
    let r = pop(stack)?;
    stack.push(Value::ValReal(-get_real(&r)));
    Ok(())
}

fn eval_real(stack: &mut Stack) -> Result<(), EvalError> {
    let i = pop(stack)?;
    stack.push(Value::ValReal(get_integer(&i) as f64));
    Ok(())
}

fn eval_sin(stack: &mut Stack) -> Result<(), EvalError> {
    let r = pop(stack)?;
    let mut res = get_real(&r).to_radians().sin();
    if res.abs() < 1e-15 {
        res = 0.0;
    }
    stack.push(Value::ValReal(res));
    Ok(())
}

fn eval_sqrt(stack: &mut Stack) -> Result<(), EvalError> {
    let r = pop(stack)?;
    stack.push(Value::ValReal(get_real(&r).sqrt()));
    Ok(())
}

fn eval_subi(stack: &mut Stack) -> Result<(), EvalError> {
    let i2 = pop(stack)?;
    let i1 = pop(stack)?;
    stack.push(Value::ValInteger(get_integer(&i1) - get_integer(&i2)));
    Ok(())
}

fn eval_subf(stack: &mut Stack) -> Result<(), EvalError> {
    let r2 = pop(stack)?;
    let r1 = pop(stack)?;
    stack.push(Value::ValReal(get_real(&r1) - get_real(&r2)));
    Ok(())
}

fn eval_point(stack: &mut Stack) -> Result<(), EvalError> {
    let z = pop(stack)?;
    let y = pop(stack)?;
    let x = pop(stack)?;
    stack.push(Value::ValPoint([get_real(&x), get_real(&y), get_real(&z)]));
    Ok(())
}

fn eval_getx(stack: &mut Stack) -> Result<(), EvalError> {
    let p = pop(stack)?;
    stack.push(Value::ValReal(get_point_x(&p)));
    Ok(())
}

fn eval_gety(stack: &mut Stack) -> Result<(), EvalError> {
    let p = pop(stack)?;
    stack.push(Value::ValReal(get_point_y(&p)));
    Ok(())
}

fn eval_getz(stack: &mut Stack) -> Result<(), EvalError> {
    let p = pop(stack)?;
    stack.push(Value::ValReal(get_point_z(&p)));
    Ok(())
}

fn get_array(v: &Value) -> &Vec<Value> {
    match v {
        &Value::ValArray(ref a) => a,
        other => panic!("{:?} is not an array", other),
    }
}

fn move_lights(v: Value) -> Vec<Box<lights::Light>> {
    match v {
        Value::ValArray(a) => {
            a.into_iter().map(|v| match v {
                Value::ValLight(light) => light,
                _ => panic!("array must be lights only!")
            }).collect()
        }
        other => panic!("{:?} is not an array", other),
    }
}

fn eval_get(stack: &mut Stack) -> Result<(), EvalError> {
    let i = pop(stack)?;
    let a = pop(stack)?;
    let iv = get_integer(&i);
    let av = get_array(&a);
    if iv < 0 || iv > av.len() as i64 {
        return Err(EvalError::ArrayOutOfBounds(iv, av.len()));
    }
    stack.push(av[iv as usize].clone());
    Ok(())
}

fn eval_length(stack: &mut Stack) -> Result<(), EvalError> {
    let a = pop(stack)?;
    stack.push(Value::ValInteger(get_array(&a).len() as i64));
    Ok(())
}

fn eval_sphere(stack: &mut Stack) -> Result<(), EvalError> {
    let surface = pop(stack)?;
    stack.push(Value::ValNode(Box::new(primitives::Sphere::new(move_surface(surface)))));
    Ok(())
}

fn eval_union(stack: &mut Stack) -> Result<(), EvalError> {
    let obj1 = pop(stack)?;
    let obj2 = pop(stack)?;
    stack.push(Value::ValNode(Box::new(primitives::Operator::make_union(move_node(obj1), move_node(obj2)))));
    Ok(())
}

fn eval_intersect(stack: &mut Stack) -> Result<(), EvalError> {
    let obj1 = pop(stack)?;
    let obj2 = pop(stack)?;
    stack.push(Value::ValNode(Box::new(primitives::Operator::make_intersect(move_node(obj1), move_node(obj2)))));
    Ok(())
}

fn eval_difference(stack: &mut Stack) -> Result<(), EvalError> {
    let obj1 = pop(stack)?;
    let obj2 = pop(stack)?;
    stack.push(Value::ValNode(Box::new(primitives::Operator::make_difference(move_node(obj1), move_node(obj2)))));
    Ok(())
}

// def eval_cube(env, stack):
//     surface, stack = pop(stack)
//     return env, push(stack, primitives.Cube(get_surface(surface)))

// def eval_cylinder(env, stack):
//     surface, stack = pop(stack)
//     return env, push(stack, primitives.Cylinder(get_surface(surface)))

// def eval_cone(env, stack):
//     surface, stack = pop(stack)
//     return env, push(stack, primitives.Cone(get_surface(surface)))

fn eval_plane(stack: &mut Stack) -> Result<(), EvalError> {
    let surface = pop(stack)?;
    stack.push(Value::ValNode(Box::new(primitives::Plane::new(move_surface(surface)))));
    Ok(())
}

fn eval_translate(stack: &mut Stack) -> Result<(), EvalError> {
    let tz = pop(stack)?;
    let ty = pop(stack)?;
    let tx = pop(stack)?;
    let obj = pop(stack)?;
    match obj {
        Value::ValNode(mut node) => {
            node.translate(get_real(&tx), get_real(&ty), get_real(&tz));
            stack.push(Value::ValNode(node));
            Ok(())
        },
        other => Err(EvalError::WrongType(other))
    }
}

fn eval_scale(stack: &mut Stack) -> Result<(), EvalError> {
    let sz = pop(stack)?;
    let sy = pop(stack)?;
    let sx = pop(stack)?;
    let obj = pop(stack)?;
    match obj {
        Value::ValNode(mut node) => {
            node.scale(get_real(&sx), get_real(&sy), get_real(&sz));
            stack.push(Value::ValNode(node));
            Ok(())
        },
        other => Err(EvalError::WrongType(other))
    }
}

fn eval_uscale(stack: &mut Stack) -> Result<(), EvalError> {
    let s = pop(stack)?;
    let obj = pop(stack)?;
    match obj {
        Value::ValNode(mut node) => {
            node.uscale(get_real(&s));
            stack.push(Value::ValNode(node));
            Ok(())
        },
        other => Err(EvalError::WrongType(other))
    }
}

fn eval_rotatex(stack: &mut Stack) -> Result<(), EvalError> {
    let d = pop(stack)?;
    let obj = pop(stack)?;
    match obj {
        Value::ValNode(mut node) => {
            node.rotatex(get_real(&d));
            stack.push(Value::ValNode(node));
            Ok(())
        },
        other => Err(EvalError::WrongType(other))
    }
}

fn eval_rotatey(stack: &mut Stack) -> Result<(), EvalError> {
    let d = pop(stack)?;
    let obj = pop(stack)?;
    match obj {
        Value::ValNode(mut node) => {
            node.rotatey(get_real(&d));
            stack.push(Value::ValNode(node));
            Ok(())
        },
        other => Err(EvalError::WrongType(other))
    }
}

fn eval_rotatez(stack: &mut Stack) -> Result<(), EvalError> {
    let d = pop(stack)?;
    let obj = pop(stack)?;
    match obj {
        Value::ValNode(mut node) => {
            node.rotatez(get_real(&d));
            stack.push(Value::ValNode(node));
            Ok(())
        },
        other => Err(EvalError::WrongType(other))
    }
}

fn eval_light(stack: &mut Stack) -> Result<(), EvalError> {
    let color = pop(stack)?;
    let d = pop(stack)?;
    stack.push(Value::ValLight(Box::new(lights::DirectionalLight::new(move_point(d),
                                                                      move_point(color)))));
    Ok(())
}

fn eval_pointlight(stack: &mut Stack) -> Result<(), EvalError> {
    let color = pop(stack)?;
    let p = pop(stack)?;
    stack.push(Value::ValLight(Box::new(lights::PointLight::new(move_point(p),
                                                                move_point(color)))));
    Ok(())
}

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

fn eval_render(stack: &mut Stack) -> Result<(), EvalError> {
    let file = pop(stack)?;
    let ht = pop(stack)?;
    let wid = pop(stack)?;
    let fov = pop(stack)?;
    let depth = pop(stack)?;
    let obj = pop(stack)?;
    let lights = pop(stack)?;
    let amb = pop(stack)?;
    raytracer::render(move_point(amb),
                      move_lights(lights),
                      move_node(obj),
                      get_integer(&depth),
                      get_real(&fov),
                      get_integer(&wid),
                      get_integer(&ht),
                      get_string(&file));
    Ok(())
}

// Let's move into here, to avoid one clone
fn move_surface(v: Value) -> Rc<Box<Fn(i64, f64, f64) -> (vecmath::Vec3, f64, f64, f64)>> {
    match v {
        Value::ValClosure(env, ast) => {
            Rc::new(Box::new(move |face: i64, u: f64, v: f64| -> (vecmath::Vec3, f64, f64, f64) {
                let mut mutable_local_env = env.clone();
                let mut stack = Stack::new();
                stack.push(Value::ValInteger(face));
                stack.push(Value::ValReal(u));
                stack.push(Value::ValReal(v));
                do_evaluate(&mut mutable_local_env, &mut stack, &ast);
                let n = stack.pop().expect("empty stack");
                let ks = stack.pop().expect("empty stack");
                let kd = stack.pop().expect("empty stack");
                let sc = stack.pop().expect("empty stack");
                (move_point(sc), get_real(&kd), get_real(&ks), get_real(&n))
            }))
        }
        _ => panic!("not a closure")
    }
}

fn do_evaluate(env: &mut Env, stack: &mut Stack, ast: &[parser::AstNode]) {
    for i in 0..ast.len() {
        match &ast[i] {
            &parser::AstNode::Function(ref v) => {
                stack.push(Value::ValClosure(env.clone(), v.clone()));
            }
            &parser::AstNode::Array(ref v) => {
                let mut local_stack = Stack::new();
                do_evaluate(env, &mut local_stack, v);
                stack.push(make_array(&local_stack));
            }
            &parser::AstNode::Leaf(ref t) => {
                match t {
                    &tokenizer::Token::Integer(v) => stack.push(Value::ValInteger(v)),
                    &tokenizer::Token::Real(v) => stack.push(Value::ValReal(v)),
                    &tokenizer::Token::Boolean(v) => stack.push(Value::ValBoolean(v)),
                    &tokenizer::Token::Str(ref v) => stack.push(Value::ValString(v.clone())),
                    &tokenizer::Token::Binder(ref v) => {
                        let i = stack.pop().expect("empty stack");
                        env.insert(v.clone(), i);
                    }
                    &tokenizer::Token::Identifier(ref v) => {
                        let val = env.get(v).expect("environment missing identifier");
                        //                 if isinstance(e, primitives.Node):
                        //                     e = copy.deepcopy(e)
                        stack.push(val.clone())
                    }
                    &tokenizer::Token::Operator(ref v) => {
                        eval_op(v, stack).expect("error evaluation op");
                    }
                    token => {
                        panic!("evaluate error, unknown token: {:?}", token);
                    }
                }
            }
        }
    }
}

fn evaluate(ast: &[parser::AstNode]) -> (Env, Stack) {
    // Apparently can't call static methods on aliased types, so her goes the full name of Env
    let mut env: collections::HashMap<String, Value> = collections::HashMap::<String, Value>::new();
    let mut stack = Stack::new();
    stack.reserve(100); // Let's avoid too many allocations
    do_evaluate(&mut env, &mut stack, &ast);
    (env, stack)
}

pub fn run(gml: &str) -> (Env, Stack) {
    evaluate(&parser::parse(&tokenizer::tokenize(gml)))
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bind_integer() {
        let (env, _) = run("1 /x");
        assert_eq!(env.get("x").unwrap(), &Value::ValInteger(1));
    }

    #[test]
    fn test_bind_string() {
        let (env, _) = run(r#""apa" /x"#);
        assert_eq!(env.get("x").unwrap(), &Value::ValString("apa".to_string()));
        assert_eq!(*get_string(env.get("x").unwrap()), "apa".to_string());
    }

    #[test]
    fn test_addi() {
        let (_, stack) = run("1 2 addi");
        assert_eq!(stack, vec![Value::ValInteger(3)]);
    }

    #[test]
    fn test_addf() {
        let (_, stack) = run("1.5 2.5 addf");
        assert_eq!(stack, vec![Value::ValReal(4.0)]);
    }

    #[test]
    fn test_ref() {
        let (_, stack) = run("1 /x x x addi");
        assert_eq!(stack, vec![Value::ValInteger(2)]);
    }

    #[test]
    fn test_apply() {
        let (_, stack) = run("1 { /x x x } apply");
        assert_eq!(stack, vec![Value::ValInteger(1), Value::ValInteger(1)]);
    }

    #[test]
    fn test_if() {
        let (_, stack) = run("true { 1 } { 2 } if");
        assert_eq!(stack, vec![Value::ValInteger(1)]);
        let (_, stack) = run("false { 1 } { 2 } if");
        assert_eq!(stack, vec![Value::ValInteger(2)]);
    }

    #[test]
    fn test_eqi() {
        let (_, stack) = run("1 2 eqi");
        assert_eq!(stack, vec![Value::ValBoolean(false)]);
        let (_, stack) = run("5 5 eqi");
        assert_eq!(stack, vec![Value::ValBoolean(true)]);
    }

    #[test]
    fn test_eqf() {
        let (_, stack) = run("1.5 2.7 eqf");
        assert_eq!(stack, vec![Value::ValBoolean(false)]);
        let (_, stack) = run("5.123 5.123 eqf");
        assert_eq!(stack, vec![Value::ValBoolean(true)]);
    }

    #[test]
    fn test_lessi() {
        let (_, stack) = run("2 1 lessi");
        assert_eq!(stack, vec![Value::ValBoolean(false)]);
        let (_, stack) = run("2 2 lessi");
        assert_eq!(stack, vec![Value::ValBoolean(false)]);
        let (_, stack) = run("1 2 lessi");
        assert_eq!(stack, vec![Value::ValBoolean(true)]);
    }

    #[test]
    fn test_lessf() {
        let (_, stack) = run("2.0 1.0 lessf");
        assert_eq!(stack, vec![Value::ValBoolean(false)]);
        let (_, stack) = run("2.0 2.0 lessf");
        assert_eq!(stack, vec![Value::ValBoolean(false)]);
        let (_, stack) = run("1.0 2.0 lessf");
        assert_eq!(stack, vec![Value::ValBoolean(true)]);
    }

    #[test]
    fn test_successful_run() {
        run("false /b b { 1 } { 2 } if");
        run("4 /x 2 x addi");
        run("1 { /x x x } apply addi");
        run("{ /x x x } /dup { dup apply muli } /sq 3 sq apply");
        run("{ /x /y x y } /swap 3 4 swap apply");
        run("{ /self /n n 2 lessi { 1 } { n 1 subi self self apply n muli } if } /fact 12 fact fact \
             apply");
    }

    #[test]
    fn test_small_program() {
        for ui in 0..10 {
            for vi in 0..10 {
                let u = ui as f64;
                let v = vi as f64;
                // Implement a small program
                // Escaping { in format strings is done by adding an extra brace -> {{
                let prog = format!("1 /col1 2 /col2 {:.1} /u {:.1} /v {{ /y /x x x mulf y y mulf \
                                    addf sqrt }} /dist \n {{ \n u 0.5 subf /u v 0.5 subf /v \n u u v \
                                    dist apply divf /b \n 0.0 v lessf {{ b asin }} {{ 360.0 b asin \
                                    subf }} if 180.0 addf 30.0 divf \n floor 2 modi 1 eqi {{ col1 }} \
                                    {{ col2 }} if \n }} apply",
                                   u / 10.0,
                                   v / 10.0);
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
                assert_eq!(stack.len(), 1);
                assert_eq!(get_integer(&stack[0]), expected);
            }
        }
    }

    #[test]
    fn test_clampf() {
        let (env, _) = run("-0.4 clampf /x");
        assert_eq!(env.get("x").unwrap(), &Value::ValReal(0.0));
        let (env, _) = run("1.1 clampf /x");
        assert_eq!(env.get("x").unwrap(), &Value::ValReal(1.0));
        let (env, _) = run("0.8 clampf /x");
        assert_eq!(env.get("x").unwrap(), &Value::ValReal(0.8));
    }

    #[test]
    fn test_array() {
        let (env, _) = run("[1 2 3] length /x");
        assert_eq!(env.get("x").unwrap(), &Value::ValInteger(3));
        let (env, _) = run("[1 2 3] 1 get /x");
        assert_eq!(env.get("x").unwrap(), &Value::ValInteger(2));
    }

    #[test]
    fn test_point() {
        let (env, _) = run("1.0 2.0 3.0 point /x");
        assert_eq!(env.get("x").unwrap(), &Value::ValPoint([1.0, 2.0, 3.0]));
        let (env, _) = run("1.0 2.0 3.0 point getx /x");
        assert_eq!(env.get("x").unwrap(), &Value::ValReal(1.0));
        let (env, _) = run("1.0 2.0 3.0 point gety /x");
        assert_eq!(env.get("x").unwrap(), &Value::ValReal(2.0));
        let (env, _) = run("1.0 2.0 3.0 point getz /x");
        assert_eq!(env.get("x").unwrap(), &Value::ValReal(3.0));
    }

    #[test]
    fn test_sphere() {
        let (env, _) = run("1.0 { /x x } sphere /x");
        match env.get("x").unwrap() {
            &Value::ValNode(ref x) => assert_eq!(x.name(), "sphere"),
            _ => assert!(false),
        }
    }

    #[test]
    fn test_light() {
        let (env, _) = run("1.0 0.0 0.0 point 0.7 0.5 0.3 point light /x");
        match env.get("x").unwrap() {
            &Value::ValLight(ref x) => {
                assert_eq!(x.name(), "light");
                assert_eq!(x.get_direction([1.0, 1.0, 1.0]), ([-1.0, 0.0, 0.0], None));
                assert_eq!(x.get_intensity([1.0, 1.0, 1.0]), [0.7, 0.5, 0.3]);
            }
            _ => assert!(false),
        }
    }

    #[test]
    fn test_render() {
        let (_, _) = run(r#"1.0 0.0 0.0 point 0.7 0.5 0.3 point light /l 1.0 { /v /u /face 0.8 0.2 v point 1.0 0.2 1.0 } sphere /s 0.5 0.5 0.5 point [ l ] s 3 90.0 320 240 "eval.ppm" render"#);
    }
}
