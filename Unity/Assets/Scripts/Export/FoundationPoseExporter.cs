using System.IO;
using System.Text;
using UnityEngine;


public class FoundationPoseExporter : IPoseExporter
{
    public string MethodName => "FoundationPose";

    public void Export(string baseFolder, string imageID, Texture2D rgbTex, Texture2D depthTex, float[] camK, string sceneGtJson, string sceneGtInfoJson)
    {
        string root = Path.Combine(baseFolder, MethodName);
        foreach (var sub in new[] { "rgb", "depth", "masks", "mesh" })
            Directory.CreateDirectory(Path.Combine(root, sub));

        string fileName = imageID + ".png";
        File.WriteAllBytes(Path.Combine(root, "rgb", fileName), rgbTex.EncodeToPNG());
        File.WriteAllBytes(Path.Combine(root, "depth", fileName), depthTex.EncodeToPNG());

        var sb = new StringBuilder();

        sb.AppendFormat("{0:E18} {1:E18} {2:E18}\n", camK[0], camK[1], camK[2]);
        sb.AppendFormat("{0:E18} {1:E18} {2:E18}\n", camK[3], camK[4], camK[5]);
        sb.AppendFormat("{0:E18} {1:E18} {2:E18}", camK[6], camK[7], camK[8]);

        File.WriteAllText(Path.Combine(root, "cam_K.txt"), sb.ToString());
    }
}
