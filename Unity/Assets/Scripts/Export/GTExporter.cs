using System.IO;
using System.Globalization;
using UnityEngine;

public class GTExporter : IPoseExporter
{
    public string MethodName => "GT";

    public void Export(string baseFolder, string imageID, Texture2D rgbTex, Texture2D depthTex, float[] camK, string sceneGtJson, string sceneGtInfoJson)
    {
        var root = Path.Combine(baseFolder, MethodName);
        Directory.CreateDirectory(root);
        File.WriteAllText(Path.Combine(root, "scene_gt.json"), sceneGtJson);
        File.WriteAllText(Path.Combine(root, "scene_gt_info.json"), sceneGtInfoJson);
    }

    string BuildCameraJson(float[] k, int height, int width) =>
        string.Format(CultureInfo.InvariantCulture,
            "{{\"K\": [[{0}, {1}, {2}], [{3}, {4}, {5}], [{6}, {7}, {8}]], \"resolution\": [{9}, {10}]}}",
            k[0], k[1], k[2],
            k[3], k[4], k[5],
            k[6], k[7], k[8],
            height, width);
}
