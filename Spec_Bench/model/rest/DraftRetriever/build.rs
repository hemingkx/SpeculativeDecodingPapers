fn main() {
    println!("cargo:rerun-if-changed=libsais.c");

    let src = [
        "src/libsais/libsais.c",
    ];
    let mut builder = cc::Build::new();
    let build = builder
        .files(src.iter());
    build.compile("libsais");
}
