extern crate rust_raytracer;
extern crate getopts;

use getopts::Options;
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::path::Path;
use std::collections::HashSet;

fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} FILE [options]", program);
    print!("{}", opts.usage(&brief));
}

fn get_contents(path: &Path, defines: &mut HashSet<String>) -> Result<Vec<String>, Box<std::error::Error>> {
    let input = File::open(path)?;
    let buffered = BufReader::new(input);
    let mut contents = Vec::<String>::new();
    let mut guard = 0;
    for line_iter in buffered.lines() {
        let line = line_iter?;
        if line.starts_with("#include \"") {
            let base = path.parent().unwrap_or(Path::new(""));
            let include_path = base.join(line[10..line.len()-1].to_string());
            let mut include_contents = get_contents(&include_path, defines)?;
            contents.append(&mut include_contents);
        } else if line.starts_with("#ifndef ") {
            let id = line[8..line.len()].to_string();
            if defines.contains(&id) {
                guard = guard + 1;
            }
        } else if line.starts_with("#define ") {
            let id = line[8..line.len()].to_string();
            defines.insert(id);
        } else if line.starts_with("#endif ") {
            guard = guard - 1;
        } else if guard == 0 {
            contents.push(line);
        }
    }
    Ok(contents)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let program = args[0].clone();

    let mut opts = Options::new();
    opts.optopt("o", "", "set output file name", "NAME");
    opts.optflag("n", "no-render", "don't run the gml");
    opts.optflag("h", "help", "print this help menu");
    let matches = match opts.parse(&args[1..]) {
        Ok(m) => { m }
        Err(f) => { panic!(f.to_string()) }
    };
    if matches.opt_present("h") {
        print_usage(&program, opts);
        return;
    }

    let input = if !matches.free.is_empty() {
        matches.free[0].clone()
    } else {
        print_usage(&program, opts);
        return;
    };

    let mut defines = HashSet::<String>::new();
    if let Ok(lines) = get_contents(Path::new(&input), &mut defines) {
        let contents = lines.join("\n");
        if let Some(output) = matches.opt_str("o") {
            println!("out file: {}", output);
            std::fs::write(output, contents.clone()).expect("can't write file");
        }
        if !matches.opt_present("n") {
            rust_raytracer::render::render_gml(&contents);
        }
    }
}
