// 1) The interface every exporter must implement
using System.IO;
using UnityEngine;
public interface IPoseExporter
{
    /// <summary>
    /// Called once per capture.  You get the raw RGB/Depth textures and the computed cam_K.
    /// You should write everything you need under: baseFolder/MethodName/
    /// </summary>
    void Export(string baseFolder, string imageID, Texture2D rgbTex, Texture2D depthTex, float[] camK, string sceneGtJson, string sceneGtInfoJson);

    /// <summary>
    /// A human‐readable name (will become the subdirectory under baseFolder).
    /// </summary>
    string MethodName { get; }
}
