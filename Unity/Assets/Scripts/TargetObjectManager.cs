using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using System;
using Newtonsoft.Json;
using TMPro;


[RequireComponent(typeof(ARTrackedImageManager))]
public class TargetObjectManager : MonoBehaviour
{
    [Header("General Settings")]
    [Tooltip("List of 6 prefabs, one for each qr code named \"1\"..\"6\". Prefab.name must match the reference image name.")]
    [SerializeField] private GameObject[] markerPrefabs;

    [SerializeField] private GameObject targetObject;
    [SerializeField] private GameObject markerObject;

    [Tooltip("If true, spawned prefabs will follow the tracked image as it moves (until locked).")]
    public bool updateMode = true;


    [Tooltip("Vertical offset applied to every marker prefab.")]
    [SerializeField] private float yOffset = 0.5f;

    [Header("Target Settings")]
    [SerializeField] private float targetYOffset = 0f;
    [SerializeField] private TMP_InputField yOffsetInputField;

    [SerializeField] public ARTrackedImageManager trackedImageManager;

    private Dictionary<string, GameObject> _spawnedMarkers = new Dictionary<string, GameObject>();
    private Dictionary<string, List<Vector3>> _positionSamples = new Dictionary<string, List<Vector3>>();
    private HashSet<string> _lockedMarkers = new HashSet<string>();


    [Header("Ground Truth Capture")]
    [Tooltip("ARCameraManager for grabbing the latest CPU image (for RGB).")]
    [SerializeField] private ARCameraManager arCameraManager;
    [Tooltip("AROcclusionManager for grabbing the latest depth image.")]
    [SerializeField] private AROcclusionManager arOcclusionManager;
    [Tooltip("Shader that renders each object with a unique flat color for mask generation.")]
    [SerializeField] private Shader maskShader;

    [Tooltip("Numeric object ID to use in the GigaPose JSON for the targetObject.")]
    [SerializeField] private int targetObjId = 1;

    [Header("Rotation Offsets")]
    [SerializeField] private float xRotationOffset = 0f;
    [SerializeField] private float yRotationOffset = 0f;
    [SerializeField] private float zRotationOffset = 0f;

    void Awake()
    {
        trackedImageManager = GetComponent<ARTrackedImageManager>();
        Debug.Log("[TargetObjectManager] Awake: ARTrackedImageManager acquired.");

        if (yOffsetInputField != null)
            yOffsetInputField.onEndEdit.AddListener(OnTargetYOffsetChanged);
    }

    private void OnDestroy()
    {
        if (yOffsetInputField != null)
            yOffsetInputField.onEndEdit.RemoveListener(OnTargetYOffsetChanged);
    }

    public bool ToggleGroundTruthObject()
    {
        targetObject.SetActive(!targetObject.activeSelf);
        return targetObject.activeSelf;
    }

    public void UpdateTargetObject(string objectType)
    {
        if (objectType == string.Empty) return;

        // Find all objects with TargetObject tag including inactive ones
        GameObject[] targetObjects = Resources.FindObjectsOfTypeAll<GameObject>()
            .Where(obj => obj.CompareTag("GroundTruthObject"))
            .ToArray();

        foreach (GameObject obj in targetObjects)
        {
            if (obj.name == objectType + "Object")
            {
                Debug.Log($"[TargetObjectManager] UpdateTargetObject: {objectType} found");
                targetObject = obj;
                obj.SetActive(true);
            }
        }
    }

    public void OnTargetYOffsetChanged(string text)
    {
        if (float.TryParse(text, out var val))
        {
            Debug.Log($"[TargetObjectManager] Target Y-offset changed to {val}");
            targetYOffset = val;

            if (markerObject != null && ShouldPoseUpdate())
            {
                Debug.Log($"[TargetObjectManager] Target Y-offset Update {val}");
                var pos = markerObject.transform.position;
                pos.y = ComputeCentroidY() + targetYOffset;
                markerObject.transform.position = pos;
            }
        }
        else
        {
            Debug.LogWarning($"Invalid Y-offset: “{text}”");
        }
    }

    private float ComputeCentroidY()
    {
        if (_spawnedMarkers.Count == 0) return markerObject.transform.position.y;
        float sum = 0f;
        foreach (var go in _spawnedMarkers.Values)
            sum += go.transform.position.y;
        return sum / _spawnedMarkers.Count;
    }

    public GameObject GetTargetObject()
    {
        if (targetObject == null)
        {
            Debug.LogError("[TargetObjectManager] GetTargetObject: targetObject is null");
            return null;
        }

        return targetObject;
    }

    public void SpawnMarker(ARTrackedImage trackedImage)
    {
        string markerName = trackedImage.referenceImage.name;
        var prefab = markerPrefabs.FirstOrDefault(p => p.name == markerName);
        if (prefab == null) return;

        Vector3 spawnPos = trackedImage.transform.position + Vector3.up * yOffset;
        var go = Instantiate(prefab, spawnPos, trackedImage.transform.rotation);
        go.name = markerName;

        _spawnedMarkers[markerName] = go;
        _positionSamples[markerName] = new List<Vector3>();
        _lockedMarkers.Remove(markerName);
    }

    public void UpdateMarker(ARTrackedImage trackedImage)
    {
        string markerName = trackedImage.referenceImage.name;
        if (markerName == null) return;
        if (!_spawnedMarkers.TryGetValue(markerName, out var go)) return;
        if (_lockedMarkers.Contains(markerName)) return;
        if (!updateMode) return;

        go.transform.position = trackedImage.transform.position + Vector3.up * yOffset;
    }

    public void RemoveMarker(ARTrackedImage trackedImage)
    {
        string markerName = trackedImage.referenceImage.name;
        if (_spawnedMarkers.TryGetValue(markerName, out var go))
        {
            Destroy(go);
            _spawnedMarkers.Remove(markerName);
            _positionSamples.Remove(markerName);
            _lockedMarkers.Remove(markerName);
        }
    }

    public void UpdateTargetObjectPose()
    {
        if (_spawnedMarkers.Count == 0 || targetObject == null) return;
        if (!ShouldPoseUpdate()) return;

        Vector3 centroid = Vector3.zero;
        foreach (var go in _spawnedMarkers.Values)
            centroid += go.transform.position;
        centroid /= _spawnedMarkers.Count;

        markerObject.transform.position = centroid;
        var pos = markerObject.transform.position;
        pos.y = ComputeCentroidY() + targetYOffset;
        markerObject.transform.position = pos;


        if (_spawnedMarkers.ContainsKey("1") && _spawnedMarkers.ContainsKey("2"))
        {
            Vector3 p1 = _spawnedMarkers["1"].transform.position;
            Vector3 p2 = _spawnedMarkers["2"].transform.position;
            Vector3 forward = (p2 - p1).normalized;

            Quaternion baseRotation = Quaternion.LookRotation(forward, Vector3.up);
            Quaternion offsetRotation = Quaternion.Euler(xRotationOffset, yRotationOffset, zRotationOffset);

            markerObject.transform.rotation = baseRotation * offsetRotation;
        }
    }

    private bool ShouldPoseUpdate()
    {
        if (!_spawnedMarkers.ContainsKey("1")
            || !_spawnedMarkers.ContainsKey("2")
            || !_spawnedMarkers.ContainsKey("5")
            || !_spawnedMarkers.ContainsKey("6"))
        {
            return false;
        }
        return true;
    }

    public (float[] R, float[] t) ComputeGTPose()
    {
        // 2) Extrinsische (Unity Model→Camera)
        Matrix4x4 M_w2c = Camera.main.transform.worldToLocalMatrix;
        Matrix4x4 M_m2w = targetObject.transform.localToWorldMatrix;
        Matrix4x4 M_m2c = M_w2c * M_m2w;

        // Extract Rotation & Translation
        var R_uni = new float[9] {
            M_m2c.m00, M_m2c.m01, M_m2c.m02,
            M_m2c.m10, M_m2c.m11, M_m2c.m12,
            M_m2c.m20, M_m2c.m21, M_m2c.m22
        };
        Vector3 t_uni = M_m2c.GetColumn(3);


        // 3) Unity (LH, Y-up) → BOP (RH, Y-down, Z-forward)
        float[] R_bop = new float[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
            {
                float signRow = (r == 0 ? +1f : -1f);
                float signCol = (c == 0 ? +1f : -1f);
                R_bop[3 * r + c] = signRow * signCol * R_uni[3 * r + c];
            }
        Vector3 t_bop = new Vector3(
            +1f * t_uni.x,
            -1f * t_uni.y,
            -1f * t_uni.z
        );

        // 4) Spiegelung um X (MirrorX) in Kamera-Koordinaten
        float[] R_final = new float[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
            {
                float s = (r == 0 ? -1f : 1f);
                R_final[3 * r + c] = s * R_bop[3 * r + c];
            }
        Vector3 t_final = new Vector3(
            -t_bop.x,
             t_bop.y,
             t_bop.z
        );

        // 5) Translation in Millimeter
        var t_mm = new float[3] {
                t_final.x * 1000f,
                t_final.y * 1000f,
                t_final.z * 1000f
            };


        float[] R0 = new float[9] {
                    -R_final[0], -R_final[1], -R_final[2],
                     R_final[3], R_final[4], R_final[5],
                    -R_final[6], -R_final[7], -R_final[8],
             };
        float[] t0 = new float[3] {
                    -t_mm[0], t_mm[1], -t_mm[2]
             };

        GameObject targetObj = targetObject;
        Vector3 meshCenterLocal = targetObj.GetComponentInChildren<MeshFilter>().sharedMesh.bounds.center;

        // 1) Apply the “small” model‐space rotation of 90° about X
        PoseUtils.ApplyModelCorrection(R0, t0, meshCenterLocal, 'x', -90f, out var R1, out var t1);
        // 2) Then mirror the result across the X axis through the center
        PoseUtils.MirrorModelAxisCenter(R1, t1, meshCenterLocal, 'x', out var R2, out var t2);

        return (R2, t2);
    }

    public (string sceneGtJson, string sceneGtInfoJson) GenerateGroundTruthForTarget()
    {
        var sceneGt = new Dictionary<string, List<SceneGtEntry>>();
        {
            string key = "0";
            var poseList = new List<SceneGtEntry>();

            // 1) Compute the pose of the targetObject
            var (R2, t2) = ComputeGTPose();

            poseList.Add(new SceneGtEntry
            {
                cam_R_m2c = R2,
                cam_t_m2c = t2,
                obj_id = targetObjId
            });
            sceneGt[key] = poseList;
        }

        var sceneInfo = new Dictionary<string, List<SceneInfoEntry>>();
        var maskCam = CreateMaskCamera();

        RenderMaskOf(targetObject, maskCam, maskShader, out var maskTex);

        using (var depthImg = TryAcquireLatestCpuImage(arOcclusionManager))
        {
            ComputeMaskStats(maskTex, depthImg,
                out RectInt bboxObj,
                out RectInt bboxVisib,
                out int pxAll,
                out int pxValid,
                out int pxVisib);

            float visibFrac = pxVisib / (float)pxAll;

            var infoEntry = new SceneInfoEntry
            {
                bbox_obj = new int[] { bboxObj.x, bboxObj.y, bboxObj.width, bboxObj.height },
                bbox_visib = new int[] { bboxVisib.x, bboxVisib.y, bboxVisib.width, bboxVisib.height },
                px_count_all = pxAll,
                px_count_valid = pxValid,
                px_count_visib = pxVisib,
                visib_fract = visibFrac
            };

            sceneInfo["0"] = new List<SceneInfoEntry> { infoEntry };

            Destroy(maskTex);
            Destroy(maskCam.gameObject);

            string jsonSceneGt = JsonConvert.SerializeObject(sceneGt, Formatting.Indented);
            string jsonSceneGtInfo = JsonConvert.SerializeObject(sceneInfo, Formatting.Indented);

            return (jsonSceneGt, jsonSceneGtInfo);
        }
    }


    private XRCpuImage TryAcquireLatestCpuImage(AROcclusionManager mgr)
    {
        if (mgr.TryAcquireEnvironmentDepthCpuImage(out var img))
            return img;
        throw new InvalidOperationException("Could not grab depth image");
    }

    private Camera CreateMaskCamera()
    {
        var go = new GameObject("MaskCam");
        var cam = go.AddComponent<Camera>();
        cam.CopyFrom(Camera.main);
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = Color.black;
        cam.cullingMask = 1 << LayerMask.NameToLayer("MaskOnly");

        cam.enabled = false;
        return cam;
    }

    private void RenderMaskOf(GameObject obj, Camera cam, Shader flatShader, out Texture2D outTex)
    {
        var prevMaterials = new List<Material>();
        var renderers = obj.GetComponentsInChildren<Renderer>();

        foreach (var r in renderers)
        {
            prevMaterials.AddRange(r.sharedMaterials);
            r.sharedMaterial = new Material(flatShader)
            {
                color = UnityEngine.Random.ColorHSV()
            };
        }

        var rt = RenderTexture.GetTemporary(Screen.width, Screen.height, 24, RenderTextureFormat.ARGB32);
        cam.targetTexture = rt;
        cam.Render();

        outTex = new Texture2D(rt.width, rt.height, TextureFormat.RGBA32, false);
        RenderTexture.active = rt;
        outTex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        outTex.Apply();

        RenderTexture.active = null;
        cam.targetTexture = null;
        RenderTexture.ReleaseTemporary(rt);

        int mi = 0;
        foreach (var r in renderers)
        {
            var mats = prevMaterials.Skip(mi).Take(r.sharedMaterials.Length).ToArray();
            r.sharedMaterials = mats;
            mi += r.sharedMaterials.Length;
        }
    }

    private void ComputeMaskStats(
    Texture2D mask,
    XRCpuImage depthImage,
    out RectInt bboxObj,
    out RectInt bboxVisib,
    out int pxAll,
    out int pxValid,
    out int pxVisib)
    {
        int w = mask.width, h = mask.height;
        bboxObj = new RectInt(w, h, 0, 0);
        bboxVisib = new RectInt(w, h, 0, 0);

        pxAll = pxValid = pxVisib = 0;
        Color32[] cols = mask.GetPixels32();
        var depthBuffer = GetDepthBuffer(depthImage, w, h);

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int idx = y * w + x;
                var c = cols[idx];
                bool objPixel = (c.r | c.g | c.b) != 0;
                if (!objPixel) continue;

                pxAll++;
                bboxObj = Encapsulate(bboxObj, x, y);

                float d = depthBuffer[idx];
                bool validDepth = d > 0f && d < 10f;
                if (validDepth)
                {
                    pxValid++;
                    bboxVisib = Encapsulate(bboxVisib, x, y);
                    pxVisib++;
                }
            }
        }
    }

    private float[] GetDepthBuffer(XRCpuImage depthImage, int targetW, int targetH)
    {
        var format = UnityEngine.TextureFormat.RFloat;

        var conversionParams = new XRCpuImage.ConversionParams
        {
            inputRect = new RectInt(0, 0, depthImage.width, depthImage.height),
            outputDimensions = new Vector2Int(depthImage.width, depthImage.height),
            outputFormat = TextureFormat.RFloat,
            transformation = XRCpuImage.Transformation.None
        };

        var rawSize = depthImage.GetConvertedDataSize(conversionParams);

        byte[] byteArray = new byte[rawSize];
        var handle = System.Runtime.InteropServices.GCHandle.Alloc(byteArray,
                                             System.Runtime.InteropServices.GCHandleType.Pinned);
        try
        {
            IntPtr addr = handle.AddrOfPinnedObject();
            depthImage.Convert(conversionParams, addr, rawSize);
        }
        finally
        {
            handle.Free();
            depthImage.Dispose();
        }
        var tex = new Texture2D(depthImage.width, depthImage.height, TextureFormat.RFloat, false);
        tex.LoadRawTextureData(byteArray);
        tex.Apply();

        float[] buf = new float[targetW * targetH];
        var px = tex.GetPixelData<float>(0);
        if (tex.width == targetW && tex.height == targetH)
        {
            px.CopyTo(buf);
        }
        else
        {
            for (int y = 0; y < targetH; y++)
                for (int x = 0; x < targetW; x++)
                {
                    int sx = Mathf.Clamp(Mathf.RoundToInt((float)x / targetW * tex.width), 0, tex.width - 1);
                    int sy = Mathf.Clamp(Mathf.RoundToInt((float)y / targetH * tex.height), 0, tex.height - 1);
                    buf[y * targetW + x] = px[sy * tex.width + sx];
                }
        }

        Destroy(tex);
        return buf;
    }


    private RectInt Encapsulate(RectInt box, int x, int y)
    {
        int xMin = Mathf.Min(box.x, x);
        int xMax = Mathf.Max(box.x + box.width, x);
        int yMin = Mathf.Min(box.y, y);
        int yMax = Mathf.Max(box.y + box.height, y);
        return new RectInt(xMin, yMin, xMax - xMin, yMax - yMin);
    }

    [Serializable]
    private struct SceneGtEntry
    {
        public float[] cam_R_m2c;
        public float[] cam_t_m2c;
        public int obj_id;
    }

    [Serializable]
    private struct SceneInfoEntry
    {
        public int[] bbox_obj;
        public int[] bbox_visib;
        public int px_count_all;
        public int px_count_valid;
        public int px_count_visib;
        public float visib_fract;
    }
}
