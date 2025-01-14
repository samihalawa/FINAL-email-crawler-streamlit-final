{pkgs}: {
  deps = [
    pkgs.postgresql
    pkgs.libyaml
    pkgs.ffmpeg-full
    pkgs.libxcrypt
    pkgs.gdb
    pkgs.glibcLocales
  ];
}
