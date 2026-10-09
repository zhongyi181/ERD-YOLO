# Third-party notices

## Ultralytics

- Source: https://github.com/ultralytics/ultralytics
- Base commit: `dfbb343547070756b68b156f4471acd1f7e01b7e`
- Framework version: 8.4.33
- License: GNU Affero General Public License v3; the full inherited text is preserved in `LICENSE`.

The base detector, native RepConv and its fusion implementation, and training framework are third-party source. The derivative retains the inherited AGPL license and existing source headers. `patches/repp3_upstream.patch` records the RepConv parser registration needed by the student network. Project method files retain their existing internal names.

## Dependencies

PyTorch, torchvision, NumPy, OpenCV, Pillow, SciPy, scikit-image, imagecorruptions, ONNX, CUDA, and TensorRT are not redistributed as installed binary dependencies. Follow their respective licenses and notices when obtaining or redistributing them.

Fog, motion blur, Gaussian noise, and contrast use the existing imagecorruptions implementation. Low light is a separate gamma transformation. Neither these third-party functions nor the base detector are claimed as original project algorithms.

## Data and comparison models

KITTI and BDD100K images, labels, and derived corrupted images are not included. Split lists contain filenames only. Obtain datasets from their providers and observe their terms.

Optional RT-DETR evaluation adapters refer to separately supplied model code and configuration. A script interface does not imply those external assets or trained models are included or authored by this project.
