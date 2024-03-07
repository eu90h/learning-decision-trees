let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
    python3.pkgs.pip
    python3.pkgs.numpy
    python3.pkgs.jupyterlab
  ];
  shellHook = ''
    # Tells pip to put packages into $PIP_PREFIX instead of the usual locations.
    # See https://pip.pypa.io/en/stable/user_guide/#environment-variables.
    export PIP_PREFIX=$(pwd)/_build/pip_packages
    export PYTHONPATH="$PIP_PREFIX/${pkgs.python3.sitePackages}:$PYTHONPATH"
    export PATH="$PIP_PREFIX/bin:$PATH"
    unset SOURCE_DATE_EPOCH
  '';
}
