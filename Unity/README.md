# PoseLab Unity Project

**Unity Version:** 6000.0.25f1

This project is based on the Bachlorthesis of Dennis Zink for 6DoF Pose estimation

## Requirements

- **Unity** 6000.0.25f1
- **AR Foundation** (ARKit/ARCore)
- **AR Occlusion** package (for depth capture)
- **Vuforia Engine** (optional, for object‐tracking ground truth)

---

## Project Setup

1. **Clone or unzip** this repository into your `Unity/Projects` folder.
2. **Open** in Unity Hub, selecting **6000.0.25f1**.
3. Let Unity import all packages and compile.

---

## Vuforia Object Tracking (Optional)

If you want to use Vuforia to track 3D objects (instead of or in addition to ARTrackedImage):

1. In **Window → Vuforia Configuration**, paste your **Vuforia License Key** into the **App License Key** field.
2. Enable **Object Recognition** and add your `.unitypackage` or `.dat` 3D Object targets under **Target Database**.
3. Create a new **VuforiaBehaviour** in your scene (e.g. ARCamera → Add Component → VuforiaBehaviour).
4. Assign your object targets to the **ObjectTracker** in the Vuforia configuration.

> **Without a valid key**, Vuforia will not initialize – you’ll only get AR Foundation marker detection.

---

## QR Code Tracking Alternative

If you don’t have Vuforia or prefer QR‐based ground truth:

1. In the **Project** window, navigate to  
   `Assets > ImageTracking > Images`
2. You’ll find pre‐generated QR codes for each sample object.
3. **Print** these QR codes at the desired physical size on paper.
4. At runtime, AR Foundation will detect them as `XRTrackedImage` entries.

---

## How to Run

1. **Build Settings** → switch to **iOS** or **Android**, depending on your target device.
2. Ensure **ARCore/ARKit** support is checked.
3. Click **Build and Run** on your device.
4. Grant camera & depth permissions when prompted.

---
