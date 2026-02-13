# shell.nix
let
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/2db38e08fdadcc0ce3232f7279bab59a15b94482.tar.gz") {};
in pkgs.mkShell {
  packages = [
	(pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
	  pytest
	]))
  ];
}
