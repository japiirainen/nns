{
  description = "python310 devenv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-22.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs
    , flake-utils
    , ...
    }:

    flake-utils.lib.eachDefaultSystem (system:
    let
      overlays = [
        (_: super: {
          python = super.python310;
        })
      ];

      pkgs = import nixpkgs { inherit overlays system; };
    in
    {
      devShells.default = pkgs.mkShell {
        packages = with pkgs; [
          python
          pyright
          black
          pylint
        ] ++
        (with pkgs.python310Packages; [
          vulture
        ]);

        shellHook = ''
          ${pkgs.python}/bin/python --version
        '';
      };
    });
}
