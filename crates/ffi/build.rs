//! Build script for the `synapse-ffi` crate.
//!
//! `llama-cpp-sys-2` builds `llama.cpp` with CMake and bundles the `llama`
//! translation units into its rlib, but the `ggml` translation units end up in
//! separate static archives (`ggml.a`, `ggml-base.a`, `ggml-cpu.a`).
//!
//! On the `win32-mingw` target `llama.cpp` intentionally removes the `lib`
//! prefix from those archives (see `ggml/CMakeLists.txt`), while the linker
//! only looks for `lib*.a` when a library is requested by name. The archives
//! are therefore mirrored under their conventional names here, and the
//! directory is published as a native library search path.
//!
//! The archives themselves are attached to the crate with the `#[link]`
//! attributes in `src/lib.rs`, so that every binary that depends on this
//! crate links them without any extra build script of its own.

use std::path::{Path, PathBuf};

/// Archives of the `ggml` library that have to be linked.
const GGML_ARCHIVES: [&str; 3] = ["ggml", "ggml-base", "ggml-cpu"];

/// Directory name prefix of the `llama-cpp-sys-2` build script output.
const SYS_CRATE_PREFIX: &str = "llama-cpp-sys-2-";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let Some(out_dir) = std::env::var_os("OUT_DIR").map(PathBuf::from) else {
        return;
    };

    // `<target>/debug/build/synapse-ffi-<hash>/out` -> `<target>/debug/build`
    let Some(build_root) = out_dir.parent().and_then(Path::parent) else {
        return;
    };

    let Some(lib_dir) = find_ggml_lib_dir(build_root) else {
        return;
    };

    let mirror_dir = out_dir.join("ggml-mirror");
    if std::fs::create_dir_all(&mirror_dir).is_err() {
        return;
    }

    let mut mirrored = false;
    for name in GGML_ARCHIVES {
        let source = lib_dir.join(format!("{name}.a"));
        if !source.is_file() {
            continue;
        }

        let target = mirror_dir.join(format!("lib{name}.a"));
        if !target.is_file() && std::fs::copy(&source, &target).is_err() {
            continue;
        }

        mirrored = true;
    }

    if mirrored {
        println!("cargo:rustc-link-search=native={}", mirror_dir.display());
    }
}

/// Finds the directory that holds the installed `ggml` archives.
///
/// Several build script output directories can exist for different feature
/// sets or crate versions, so the most recently written one is used.
fn find_ggml_lib_dir(build_root: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(build_root).ok()?;

    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in entries.flatten() {
        let path = entry.path();
        let is_sys_crate = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with(SYS_CRATE_PREFIX));
        if !is_sys_crate {
            continue;
        }

        let lib_dir = path.join("out").join("lib");
        if !lib_dir.join("ggml.a").is_file() {
            continue;
        }

        let modified = entry
            .metadata()
            .and_then(|metadata| metadata.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

        let is_newer = match best.as_ref() {
            Some((best_modified, _)) => modified > *best_modified,
            None => true,
        };
        if is_newer {
            best = Some((modified, lib_dir));
        }
    }

    best.map(|(_, lib_dir)| lib_dir)
}
