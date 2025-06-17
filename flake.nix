{
  description = "OpenCV + Python + GTK + GStreamer devshell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forEachSystem = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forEachSystem (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
          };

          opencvCustom = pkgs.opencv.override {
            enablePython = true;
            enableGtk3 = true;
            enableContrib = true;
          };

          # Use cv2 from GTK-enabled OpenCV
          cv2Python = pkgs.python3Packages.buildPythonPackage {
            pname = "cv2";
            version = opencvCustom.version;
            src = opencvCustom;
            format = "other";
            nativeBuildInputs = [ pkgs.makeWrapper ];
            installPhase = ''
              mkdir -p $out/${pkgs.python3.sitePackages}
              ln -s ${opencvCustom}/lib/python3.*/site-packages/cv2 $out/${pkgs.python3.sitePackages}/cv2
            '';
          };

          pythonEnv = pkgs.python3.withPackages (
            ps: with ps; [
              numpy
              scikit-learn
              matplotlib
              cv2Python
            ]
          );
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              opencvCustom
              gtk3
              pythonEnv

              # GStreamer
              gst_all_1.gstreamer
              gst_all_1.gst-plugins-base
              gst_all_1.gst-plugins-good
              gst_all_1.gst-plugins-bad
              gst_all_1.gst-plugins-ugly
              gst_all_1.gst-libav
            ];

            shellHook = ''
              export LD_LIBRARY_PATH=${pkgs.gtk3}/lib:${pkgs.gst_all_1.gstreamer}/lib:$LD_LIBRARY_PATH
              echo "✔️ Devshell with GTK-enabled cv2 ready"
            '';
          };
        }
      );

      nixosModules.default =
        { pkgs, ... }:
        {
          environment.systemPackages = with pkgs; [
            opencv
            gtk3
            gst_all_1.gstreamer
            gst_all_1.gst-plugins-base
            gst_all_1.gst-plugins-good
            gst_all_1.gst-plugins-bad
            gst_all_1.gst-plugins-ugly
            gst_all_1.gst-libav
          ];
        };
    };
}
