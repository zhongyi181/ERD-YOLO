# Environment

The vendored framework identifies itself as Ultralytics 8.4.33 and is based on commit `dfbb343547070756b68b156f4471acd1f7e01b7e`. Framework dependencies are declared in `pyproject.toml`.

The offline corruption tools additionally depend on imagecorruptions 1.1.2, NumPy, Pillow, SciPy, and scikit-image; motion blur needs Wand/ImageMagick.

Install PyTorch, CUDA, and optional export dependencies for your own system. Jetson packages must match the installed JetPack environment. No Conda environment, GPU runtime, or model binary is bundled. This release supplies no claim that a specific environment reproduces the manuscript metrics.
