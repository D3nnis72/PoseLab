using System;
using System.IO;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using NativeFilePickerNamespace;
using System.Collections;
using System.Linq;
using TMPro;

[Serializable]
public class PoseResponse { public float[] position; public float[] rotation; }
[Serializable]
public class CaptureHeader
{
    public float[] camK;
    public string sceneGtJson;
    public string sceneGtInfoJson;
    public string imageID;
    public string objectType;
    public string sceneName;
}


public class CaptureManager : MonoBehaviour
{
    public ARCameraManager cameraManager;
    public AROcclusionManager occlusionManager;
    public Button captureButton;
    public Button closeSocketButton;
    public bool webSocketMode = false;
    public bool mockMode = false;
    public GameObject poseTarget;
    public int targetWidth = 640;
    public int targetHeight = 480;
    public string serverUrl = "ws://100.69.178.95";

    private string _currentWebSocketModel = "";

    public TargetObjectManager grundTruthPoseEstimator;

    private string baseFolder;
    private int imageIndex;
    private ClientWebSocket ws;
    private CancellationTokenSource cts;
    private readonly XRCpuImage.Transformation imageTransform = XRCpuImage.Transformation.MirrorX;
    private List<IPoseExporter> exporters;
    private delegate bool AcquireImage(out XRCpuImage image);

    private bool _isWebsocketConnected = false;

    [Header("Buttons")]
    [SerializeField] private Button dataModeButton;
    [SerializeField] private Button hideGroundTruthObjectButton;
    private readonly Color _primaryColor = HexToColor("B4C6E4");
    private readonly Color _accentColor = HexToColor("444444");



    async void Start()
    {
        captureButton.onClick.AddListener(CaptureImages);
        closeSocketButton.onClick.AddListener(CloseWebSocket);
        SetupFolder();
        exporters = new List<IPoseExporter> { new Sam6DExporter(), new FoundationPoseExporter(), new MegaPoseExporter(), new GTExporter() };
        closeSocketButton.gameObject.SetActive(webSocketMode);


        occlusionManager.requestedEnvironmentDepthMode = EnvironmentDepthMode.Best;
        dataModeButton.onClick.AddListener(() => ToggleDataMode());
        hideGroundTruthObjectButton.onClick.AddListener(() => ToggleGroundTruthObject());
        ToggleDataMode();

    }

    void ToggleDataMode()
    {
        if (webSocketMode)
        {
            webSocketMode = false;
            dataModeButton.GetComponentInChildren<TextMeshProUGUI>().text = "File export";
        }
        else
        {
            webSocketMode = true;
            dataModeButton.GetComponentInChildren<TextMeshProUGUI>().text = "WebSocket";
        }
        SetButtonColor(dataModeButton, webSocketMode ? _accentColor : _primaryColor);

        Debug.Log($"[CaptureManager] WebSocket mode: {webSocketMode}");
    }

    void ToggleGroundTruthObject()
    {
        SetButtonColor(hideGroundTruthObjectButton, grundTruthPoseEstimator.ToggleGroundTruthObject() ? _accentColor : _primaryColor);
    }

    private void SetButtonColor(Button btn, Color col)
    {

        var img = btn.GetComponent<Image>();
        if (img != null) img.color = col;
    }


    private static Color HexToColor(string hex)
    {
        if (ColorUtility.TryParseHtmlString("#" + hex, out var c))
            return c;
        return Color.white;
    }

    public void UpdateTargetObject()
    {
        string objectType = UIManager.Instance.CadModel;

        if (objectType == string.Empty) return;

        GameObject[] targetObjects = Resources.FindObjectsOfTypeAll<GameObject>()
            .Where(obj => obj.CompareTag("PoseTargetObject"))
            .ToArray();

        foreach (GameObject obj in targetObjects)
        {
            if (obj.name == objectType + "Object")
            {
                poseTarget = obj;
                obj.SetActive(true);
            }
            else
            {
                obj.SetActive(false);
            }
        }
    }

    async Task ConnectWebSocket()
    {
        cts = new CancellationTokenSource();
        ws = new ClientWebSocket();
        string completeUrl = serverUrl + ":" + GetPortName();
        Debug.Log("completeUrl: " + completeUrl);
        await ws.ConnectAsync(new Uri(completeUrl), cts.Token);
        Debug.Log("WebSocket connected to " + completeUrl);
        _isWebsocketConnected = true;
    }

    private string GetPortName()
    {
        var model = UIManager.Instance.PoseModel;
        Debug.Log("Pose model: " + model);
        switch (model)
        {
            case "OVE6D":
                return "8765";
            case "MegaPose":
                return "8764";
            case "GigaPose":
                return "8761";
            case "FoundationPose":
                return "8763";
            case "SAM6D":
                return "8762";
            default:
                return "UnknownPort";
        }
    }


    async void CaptureImages()
    {
        if (webSocketMode && !_isWebsocketConnected)
        {
            await ConnectWebSocket();
        }
        else if (webSocketMode && _isWebsocketConnected && ws.State != WebSocketState.Open)
        {
            CloseWebSocket();
            await ConnectWebSocket();
        }

        Texture2D rgb, depth;
        float[] camK;
        string id = imageIndex.ToString("D6");
        var (sceneGtJson, sceneGtInfoJson) = grundTruthPoseEstimator.GenerateGroundTruthForTarget();

        if (mockMode)
        {
            rgb = await CaptureFrameAsync();
            depth = GenerateMockDepth();
            camK = new[] { 1f, 0, 0, 0, 1f, 0, 0, 0, 1f };
        }
        else
        {
            if (!cameraManager.TryAcquireLatestCpuImage(out var imgR)) return;
            rgb = ConvertImage(imgR, false);
            imgR.Dispose();

            if (!occlusionManager.TryAcquireRawEnvironmentDepthCpuImage(out var imgD))
                return;

            Texture2D depthFloat = ConvertImage(imgD, depth: true, skipToDepth16: true);

            int cx = depthFloat.width / 2;
            int cy = depthFloat.height / 2;
            float distMeters = depthFloat.GetPixel(cx, cy).r;
            Debug.Log($"LiDAR native depth = {distMeters:F2} m");

            depth = ToDepth16(depthFloat);

            imgD.Dispose();

            camK = ComputeCamK();
        }

        Debug.Log("Capturing image 4");
        if (webSocketMode)
        {
            Debug.Log("Capturing image WebSocket");

            var camTransform = Camera.main.transform;

            Matrix4x4 camToWorld = camTransform.localToWorldMatrix;

            String sceneName = UIManager.Instance.SceneName;
            if (string.IsNullOrEmpty(sceneName))
            {
                sceneName = DateTime.UtcNow.ToString("yyyyMMdd'T'HHmmssfff");
            }

            var resp = await SendWebSocketAndGetPose(rgb, depth, camK, id, sceneGtJson, sceneGtInfoJson, sceneName);
            if (resp != null)
            {
                ApplyPose(resp, camToWorld);
            }
        }
        else
        {
            Debug.Log("Capturing image Export");
            foreach (var e in exporters) e.Export(baseFolder, id, rgb, depth, camK, sceneGtJson, sceneGtInfoJson);
            NativeFilePicker.ExportMultipleFiles(new[] { baseFolder }, _ => { });
        }
    }

    Texture2D GenerateMockDepth()
    {
        Debug.Log("GenerateMockDepth");
        var dst = new Texture2D(targetWidth, targetHeight, TextureFormat.R16, false);
        var data = new ushort[targetWidth * targetHeight];
        var rnd = new System.Random();
        for (int i = 0; i < data.Length; i++) data[i] = (ushort)rnd.Next(0, 1000);
        dst.SetPixelData(data, 0);
        dst.Apply();
        return dst;
    }

    private Task<Texture2D> CaptureFrameAsync()
    {
        var tcs = new TaskCompletionSource<Texture2D>();
        StartCoroutine(CaptureFrameRoutine(tcs));
        return tcs.Task;
    }

    private IEnumerator CaptureFrameRoutine(TaskCompletionSource<Texture2D> tcs)
    {
        yield return new WaitForEndOfFrame();
        var tex = ScreenCapture.CaptureScreenshotAsTexture();
        tcs.SetResult(tex);
    }

    async Task<PoseResponse> SendWebSocketAndGetPose(
            Texture2D rgb, Texture2D depth, float[] camK, string id, string sceneGtJson, string sceneGtInfoJson, string sceneName)
    {
        var objectType = UIManager.Instance.CadModel;

        var hdr = new CaptureHeader { camK = camK, sceneGtJson = sceneGtJson, sceneGtInfoJson = sceneGtInfoJson, imageID = id, objectType = objectType, sceneName = sceneName };
        var hdrJson = JsonUtility.ToJson(hdr);
        var hdrBytes = Encoding.UTF8.GetBytes(hdrJson);
        await ws.SendAsync(
            new ArraySegment<byte>(hdrBytes),
            WebSocketMessageType.Text, true, cts.Token);

        await ws.SendAsync(
            new ArraySegment<byte>(rgb.EncodeToJPG()),
            WebSocketMessageType.Binary, true, cts.Token);
        await ws.SendAsync(
            new ArraySegment<byte>(depth.EncodeToPNG()),
            WebSocketMessageType.Binary, true, cts.Token);

        var buf = new ArraySegment<byte>(new byte[16 * 1024]);
        using var ms = new MemoryStream();
        WebSocketReceiveResult res;
        do
        {
            res = await ws.ReceiveAsync(buf, cts.Token);
            ms.Write(buf.Array, buf.Offset, res.Count);
        } while (!res.EndOfMessage);

        if (res.MessageType == WebSocketMessageType.Text)
        {
            var text = Encoding.UTF8.GetString(ms.ToArray());
            return JsonUtility.FromJson<PoseResponse>(text);
        }
        return null;
    }

    public void TestApplyPose()
    {
        var (R2, t2) = grundTruthPoseEstimator.ComputeGTPose();
        var resp = new PoseResponse
        {
            position = t2,
            rotation = R2
        };
        var camTransform = Camera.main.transform;
        Matrix4x4 M_c2w = camTransform.localToWorldMatrix;
        ApplyPose(resp, M_c2w);
    }

    void ApplyPose(PoseResponse resp, Matrix4x4 camTransform)
    {
        UpdateTargetObject();
        Vector3 meshCenterLocal = poseTarget.GetComponentInChildren<MeshFilter>().sharedMesh.bounds.center;
        PoseApplier.ApplyPose(resp, camTransform, poseTarget.transform, meshCenterLocal);
    }

    public async void CloseWebSocket()
    {
        if (!_isWebsocketConnected)
        {
            Debug.LogWarning("WebSocket is not connected, cannot close.");
            return;
        }

        cts.Cancel();
        if (ws != null)
        {
            await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "bye", CancellationToken.None);
            ws.Dispose();
            _isWebsocketConnected = false;
        }
    }

    Texture2D ConvertImage(XRCpuImage img, bool depth, bool skipToDepth16 = false)
    {
        var fmt = depth ? TextureFormat.RFloat : TextureFormat.RGBA32;
        var tex = new Texture2D(img.width, img.height, fmt, false);
        img.Convert(new XRCpuImage.ConversionParams
        {
            inputRect = new RectInt(0, 0, img.width, img.height),
            outputDimensions = new Vector2Int(img.width, img.height),
            outputFormat = fmt,
            transformation = imageTransform
        }, tex.GetRawTextureData<byte>());
        tex.Apply();

        if (depth && skipToDepth16)
        {
            if (tex.width != targetWidth || tex.height != targetHeight)
                tex = ResizeRFloat(tex);
            return tex;
        }

        if (tex.width != targetWidth || tex.height != targetHeight)
        {
            Debug.Log($"Resizing {tex.width}x{tex.height} to {targetWidth}x{targetHeight}");
            tex = Resize(tex, targetWidth, targetHeight);
        }
        else
        {
            Debug.Log($"Keeping {tex.width}x{tex.height}");
        }

        if (depth && !skipToDepth16)
            return ToDepth16(tex);
        return tex;
    }

    Texture2D Resize(Texture2D src, int w, int h)
    {
        var rt = RenderTexture.GetTemporary(w, h);
        Graphics.Blit(src, rt);
        var dst = new Texture2D(w, h, src.format, false);
        RenderTexture.active = rt;
        dst.ReadPixels(new Rect(0, 0, w, h), 0, 0);
        dst.Apply();
        RenderTexture.ReleaseTemporary(rt);
        return dst;
    }

    Texture2D ResizeRFloat(Texture2D src)
    {
        var rt = RenderTexture.GetTemporary(
            targetWidth, targetHeight, 0,
            RenderTextureFormat.RFloat,
            RenderTextureReadWrite.Linear
        );

        Graphics.Blit(src, rt);

        var dst = new Texture2D(targetWidth, targetHeight, TextureFormat.RFloat, false);
        RenderTexture.active = rt;
        dst.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        dst.Apply();

        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);
        return dst;
    }


    Texture2D ToDepth16(Texture2D src)
    {
        int w = src.width, h = src.height;
        var dst = new Texture2D(w, h, TextureFormat.R16, false);
        var px = src.GetPixels();
        var data = new ushort[w * h];
        for (int i = 0; i < px.Length; i++)
            data[i] = (ushort)Mathf.Clamp(Mathf.RoundToInt(px[i].r * 1000f), 0, 65535);
        dst.SetPixelData(data, 0);
        dst.Apply();
        return dst;
    }

    float[] ComputeCamK()
    {
        if (cameraManager.TryGetIntrinsics(out var intr))
        {
            float sx = (float)targetWidth / intr.resolution.x;
            float sy = (float)targetHeight / intr.resolution.y;
            float fx = intr.focalLength.x * sx;
            float fy = intr.focalLength.y * sy;
            float cx = intr.principalPoint.x * sx;
            float cy = intr.principalPoint.y * sy;
            if (imageTransform == XRCpuImage.Transformation.MirrorX)
                cx = (targetWidth - 1) - cx;
            return new[] { fx, 0, cx, 0, fy, cy, 0, 0, 1f };
        }
        return new[] { 572.4114f, 0, 325.2611f, 0, 573.5704f, 242.0490f, 0, 0, 1f };
    }

    void SetupFolder()
    {
        baseFolder = Path.Combine(Application.persistentDataPath, "BOP_dataset");
        if (Directory.Exists(baseFolder)) Directory.Delete(baseFolder, true);
        Directory.CreateDirectory(baseFolder);
    }
}
