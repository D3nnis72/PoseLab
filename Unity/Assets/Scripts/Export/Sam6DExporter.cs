using System.IO;
using UnityEngine;

[System.Serializable]
public class SceneCameraData
{
    public float[] cam_K;
    public float depth_scale;
}

public class Sam6DExporter : IPoseExporter
{
    public string MethodName => "SAM6D";

    public void Export(string baseFolder, string imageID, Texture2D rgbTex, Texture2D depthTex, float[] camK, string sceneGtJson, string sceneGtInfoJson)
    {
        string root = Path.Combine(baseFolder, MethodName);
        Directory.CreateDirectory(root);

        File.WriteAllBytes(Path.Combine(root, "rgb.png"), rgbTex.EncodeToPNG());
        File.WriteAllBytes(Path.Combine(root, "depth.png"), depthTex.EncodeToPNG());

        var camData = new SceneCameraData { cam_K = camK, depth_scale = 1.0f };
        string json = JsonUtility.ToJson(camData, true);
        File.WriteAllText(Path.Combine(root, "camera.json"), json);
    }
}
