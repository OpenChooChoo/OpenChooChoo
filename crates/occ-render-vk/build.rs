use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let shaders_dir = manifest_dir.join("shaders");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));

    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed={}", shaders_dir.display());
    emit_rerun_for_directory(&shaders_dir);

    if Command::new("slangc").arg("-v").output().is_err() {
        println!(
            "cargo::error=slangc was not found on PATH. Install Slang (https://github.com/shader-slang/slang)."
        );
        return;
    }

    let mut shaders: BTreeMap<String, PathBuf> = BTreeMap::new();
    collect_shaders(&shaders_dir, &shaders_dir, &mut shaders);

    for (rel_path, source_path) in &shaders {
        if rel_path.starts_with("common/") || rel_path.starts_with("common\\") {
            continue;
        }
        let stem = Path::new(rel_path).file_stem().unwrap().to_string_lossy().to_string();
        compile_entry(source_path, &stem, "vertexMain", "vertex", &out_dir, "vert");
        compile_entry(source_path, &stem, "fragmentMain", "fragment", &out_dir, "frag");
    }
}

fn emit_rerun_for_directory(dir: &Path) {
    let Ok(read_dir) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in read_dir.flatten() {
        let path = entry.path();
        println!("cargo::rerun-if-changed={}", path.display());
        if path.is_dir() {
            emit_rerun_for_directory(&path);
        }
    }
}

fn collect_shaders(root: &Path, dir: &Path, out: &mut BTreeMap<String, PathBuf>) {
    let Ok(read_dir) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in read_dir.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_shaders(root, &path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("slang") {
            let rel = path.strip_prefix(root).unwrap().to_string_lossy().into_owned();
            out.insert(rel, path);
        }
    }
}

fn compile_entry(
    source: &Path,
    stem: &str,
    entry: &str,
    stage: &str,
    out_dir: &Path,
    ext: &str,
) {
    let output_path = out_dir.join(format!("{stem}.{ext}.spv"));
    let status = Command::new("slangc")
        .arg(source)
        .arg("-target")
        .arg("spirv")
        .arg("-profile")
        .arg("spirv_1_6")
        .arg("-emit-spirv-directly")
        .arg("-fvk-use-entrypoint-name")
        .arg("-matrix-layout-column-major")
        .arg("-entry")
        .arg(entry)
        .arg("-stage")
        .arg(stage)
        .arg("-o")
        .arg(&output_path)
        .status();

    match status {
        Ok(status) if status.success() => {}
        Ok(status) => {
            println!(
                "cargo::error=slangc failed for {} entry {entry} (exit {status})",
                source.display()
            );
        }
        Err(err) => {
            println!(
                "cargo::error=failed to invoke slangc for {} entry {entry}: {err}",
                source.display()
            );
        }
    }
}
