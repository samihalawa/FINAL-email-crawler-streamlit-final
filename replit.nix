{pkgs}: {
  deps = [
    pkgs.python312Packages.playwright
    pkgs.postgresql
    pkgs.libyaml
    pkgs.ffmpeg-full
    pkgs.libxcrypt
    pkgs.gdb
    pkgs.glibcLocales
  ];
}
