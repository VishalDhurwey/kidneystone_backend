{ pkgs }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python312
    pkgs.python312Packages.pip
    pkgs.python312Packages.setuptools
    pkgs.libGL
    pkgs.ffmpeg
    pkgs.glib
  ];
}
