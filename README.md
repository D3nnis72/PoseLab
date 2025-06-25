# Generalizable 6DoF Pose Estimation in Augmented Reality

Prototype implementation for the bachelor thesis of **Dennis Zink** at **Hochschule der Medien** in collaboration with **Mercedes-Benz**.  
Topic: **“Generalizable 6DoF Pose Estimation in Augmented Reality: A Comparison of Modern Methods.”**

> 📄 Link to the full thesis and paper will be added soon.

---

## Abstract

Accurately estimating the 6DoF pose of novel objects is essential for applications such as augmented reality, robotics, and automated inspection, particularly in industrial environments. In the context of vehicle development at Mercedes-Benz, robust pose estimation enables dynamic use cases including real-time visualization of components during assembly and quality control.

This bachelor thesis evaluates and compares five modern deep learning-based methods for 6DoF pose estimation of previously unseen objects, using only a CAD model as input. The selected methods are FoundationPose, SAM-6D, MegaPose, GigaPose, and OVE6D. Each method was implemented within a unified, Python-based evaluation pipeline and tested on real automotive components, such as a brake disc and a crankcase, under realistic conditions.

The evaluation focused on accuracy, inference time, and generalizability, particularly in dynamic scenarios without object-specific training. The results show that FoundationPose consistently achieved the highest accuracy and robustness. MegaPose provided a solid trade-off between runtime and generalization. SAM-6D delivered promising results on simple, geometric objects but showed clear limitations when applied to more complex shapes. In contrast, GigaPose and OVE6D were affected by unstable predictions and long preprocessing times.

The thesis concludes that the tested methods are fundamentally suitable for industrial use. However, further development is required to achieve a lightweight, real-time capable all-in-one pipeline. This includes reducing model size for faster inference, integrating robust and automated segmentation, and streamlining the overall system for seamless deployment. Based on the experimental findings, this work outlines the technical and methodological improvements needed to enable a robust and real-time capable pose estimation pipeline for effective integration into augmented reality applications within the vehicle development process.

Keywords: `object pose estimation`, `6DoF`, `augmented reality`, `deep learning`, `CAD-based methods`, `computer vision`

## Folder Structure

### Root directory contains:

#### Method folders

- `SAM6D`, `gigapose`, `FoundationPose`, `megapose6d`, `SAM-6D`, `OVE6D-pose`

Instructions:

- Download and install each method manually from its official GitHub repository as described in their respective READMEs:

  - [FoundationPose](https://github.com/NVlabs/FoundationPose) — developed by Bowen Wen et al.
  - [SAM-6D](https://github.com/JiehongLin/SAM-6D) — developed by Jiehong Lin et al.
  - [MegaPose](https://github.com/megapose6d/megapose6d) — developed by Yann Labbé et al.
  - [GigaPose](https://github.com/nv-nguyen/gigapose) — developed by Vu Nguyen et al.
  - [OVE6D](https://github.com/dingdingcai/OVE6D-pose) — developed by Dingding Cai et al.

* Run their demo projects to verify that pose estimation works correctly.
* Only the pretrained weights provided by the authors were used – no retraining or object-specific fine-tuning is performed.
* Copy the evaluation scripts from this repository into the corresponding installed method folders.  
  For example: copy the scripts from the `MegaPose` folder in this repo into your local `megapose` directory.

#### Environments

All methods were tested with Conda environments:

- `sam6d`
- `megapose`
- `ove6d`
- `gigapose`

Only FoundationPose uses Docker.  
If you are using different environment names, make sure to update them in the corresponding `pipeline.py` scripts.

---

### `Result/`

Stores all generated pose estimation results and runtime logs.

---

### `SharedData/`

Contains all data required for shared evaluation across methods.

#### `ObjectmodelsM/`

Contains all object models in millimeter scale.  
Each object (e.g., `brake`, `crankcase`) contains the following folders:

- `bop_templates/` – GigaPose templates (e.g., `000001.png`)
- `model/`
  - `obj_000001.ply`
  - `models_info.json`  
    (if missing, use `calculate_model_info.py` from the `Debug` folder)
- `object_poses/` – GigaPose pose matrices (`000001.npy`, created by `render_bop_templates`)
- `ove6d_codebook/` – copied from `OVE6D-pose/evaluation/object_codebooks`
- `segmentation_templates/` – BlenderProc-rendered templates for SAM6D

💡 To verify depth quality, use `validate_depth_data.py` from the `Debug` folder.

---

### Integration hints

- Copy `Batch_inference.py` from the `GigaPose` folder into `SAM6D/Instance_Segmentation_Model/`
- Copy `render_templates_optimized.py` into `SAM6D/Render/`  
  (Improves loading speed for `.ply` models)

---

## Evaluation Modes

### 1. Batch pipeline execution

Use `completePipeline.py` to evaluate entire datasets that follow the Unity export structure.

**Expected input structure:**

```yaml
Input/
└── model/
└── visibility/
└── illumination/
└── distance/
└── height/
└── angle/
```

Results are saved in the `Result/` directory.

---

### 2. Live inference via Unity app

- Start the WebSocket server
- Launch the Unity iPad app
- Select a method and object
- Capture an image and start pose estimation
- The results are written to the `Result/` folder

To avoid firewall or IP configuration issues, it is recommended to use [Tailscale](https://tailscale.com) to connect iPad and server.  
If not using Tailscale, manually set the correct IP addresses in the WebSocket client and server scripts.

---

## Postprocessing & Visualization

1. Run `calculateAvgResult.py` to compute averages across all test conditions
2. Use `createDetailExcel.py` and `createTableExcel.py` to generate formatted Excel summaries of the results

---

## Unity App Integration

The Unity-based capture application used for inference is included in this repository under the [`Unity/`](./Unity/) directory.

It contains:

- The iPad application used to capture RGB-D data and send it to the server
- A WebSocket-based communication interface
- Manual marker-based ground truth capture (optional)

Please refer to the [`Unity/README.md`](./Unity/README.md) file for setup instructions, build process, and configuration details.

This app enables both live pose estimation as well as data export for batch evaluation. It is designed to integrate seamlessly with the Python evaluation pipelines provided in this repository.

---

## Notes

- Double-check all paths in the pipeline scripts for your local file structure
- Make sure the correct Conda or Docker environment is activated when running inference
- All required templates and JSON files must be present before executing a method
- Unity exports must follow the expected folder hierarchy for full compatibility

### Credits and Acknowledgments

This project would not have been possible without the outstanding contributions of the following authors and research teams. All 6DoF pose estimation methods evaluated in this thesis are based on their official open-source implementations:

- [FoundationPose](https://github.com/NVlabs/FoundationPose) – developed by researchers at **NVIDIA**
- [SAM-6D](https://github.com/JiehongLin/SAM-6D) – developed by **Jiehong Lin** et al.
- [MegaPose](https://github.com/megapose6d/megapose6d) – developed by **Yann Labbé** et al.
- [GigaPose](https://github.com/nv-nguyen/gigapose) – developed by **Vu Nguyen** et al.
- [OVE6D](https://github.com/dingdingcai/OVE6D-pose) – developed by **Dingding Cai** et al.

This repository integrates these methods into a unified benchmarking and evaluation pipeline.  
All rights, credit, and recognition go to the respective authors of the methods.  
The goal of this thesis is to analyze and compare these approaches under a shared evaluation framework using real-world industrial data.

If you use or build upon this work, please also cite the original papers and repositories accordingly.

Special thanks to:

- The authors of the BOP Toolkit for their standardized evaluation framework
- The open-source community for enabling reproducible research
- Mercedes-Benz and Hochschule der Medien for supporting this project

This bachelor thesis was conducted by **Dennis Zink** at Hochschule der Medien Stuttgart  
in collaboration with **Mercedes-Benz AG**, 2025.
