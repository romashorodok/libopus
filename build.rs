use std::path::PathBuf;

extern crate bindgen;

fn generate_lib() {
  #[derive(Debug)]
  struct ParseCallbacks;

  impl bindgen::callbacks::ParseCallbacks for ParseCallbacks {
    fn int_macro(
      &self, name: &str, _value: i64,
    ) -> Option<bindgen::callbacks::IntKind> {
      if name.starts_with("OPUS") {
        Some(bindgen::callbacks::IntKind::Int)
      } else {
        None
      }
    }
  }

  const PREPEND_LIB: &'static str = "
#![no_std]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
";

  let out = PathBuf::new().join("src").join("libopus.rs");

  let bindings = bindgen::Builder::default()
    .header("src/wrapper.h")
    .raw_line(PREPEND_LIB)
    .parse_callbacks(Box::new(ParseCallbacks))
    .generate_comments(false)
    .layout_tests(false)
    .ctypes_prefix("libc")
    .allowlist_type("[oO]pus.+")
    .allowlist_function("[oO]pus.+")
    .allowlist_var("[oO].+")
    .use_core()
    .generate()
    .expect("Unable to generate bindings");

  bindings.write_to_file(out).expect("Couldn't write bindings!");
}

fn main() {
  generate_lib();

  const OPUS_DIR: &str = "opus";

  let mut cmake = cmake::Config::new(OPUS_DIR);

  cmake
    .define("OPUS_INSTALL_PKG_CONFIG_MODULE", "OFF")
    .define("OPUS_INSTALL_CMAKE_CONFIG_MODULE", "OFF")
    .define("CMAKE_INSTALL_BINDIR", "bin")
    .define("CMAKE_INSTALL_MANDIR", "man")
    .define("CMAKE_INSTALL_INCLUDEDIR", "include")
    .define("CMAKE_INSTALL_OLDINCLUDEDIR", "include")
    .define("CMAKE_INSTALL_LIBDIR", "lib")
    .define("CMAKE_TRY_COMPILE_TARGET_TYPE", "STATIC_LIBRARY");

  if std::process::Command::new("ninja")
    .arg("--version")
    .status()
    .map(|status| status.success())
    .unwrap_or(false)
  {
    cmake.generator("Ninja");
  }

  println!("cargo:rustc-link-lib=static=opus");

  let mut out_dir = cmake.build();

  out_dir.push("lib");

  println!("cargo:rustc-link-search=native={}", out_dir.display());
  #[cfg(target_os = "linux")]
  {
    out_dir.pop();
    out_dir.push("lib64");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
  }
}
