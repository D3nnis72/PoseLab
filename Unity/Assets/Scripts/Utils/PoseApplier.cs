using Newtonsoft.Json.Serialization;
using UnityEngine;


public static class PoseApplier
{
    // / <summary>
    // / Apply a BOP-format pose(rotation + translation from PoseResponse)
    // / back onto a Unity GameObject, inverting your GT-generation pipeline.
    // / </summary>
    public static void ApplyPose(PoseResponse resp, Matrix4x4 camToWorldMatrix, Transform targetTransform, Vector3 meshCenterLocal)
    {
        // 1) Invert the MirrorX step:
        PoseUtils.MirrorModelAxisCenter(
            resp.rotation,
            resp.position,
            meshCenterLocal,
            'x',
            out float[] R_afterMirror,
            out float[] t_afterMirror
        );

        // 2) Invert the +90° X-rotation correction:
        PoseUtils.ApplyModelCorrection(
            R_afterMirror,
            t_afterMirror,
            meshCenterLocal,
            'x',
            90f,
            out float[] R0,
            out float[] t0
        );

        float[] R0_1 = new float[9] {
                    R0[0], -R0[1], -R0[2],
                     -R0[3], R0[4], R0[5],
                    R0[6], -R0[7], -R0[8],
        };
        float[] t0_1 = new float[3] {
                    t0[0], -t0[1], t0[2]
        };

        // 3) Build the 4×4 model→camera matrix from R0, t0.
        //    Note: t0 is still in millimeters → convert to meters here.
        Matrix4x4 M_m2c = MatrixFromR_T(R0_1, new Vector3(t0_1[0], t0_1[1], t0_1[2]) * 0.001f);

        // 4) Invert to get camera→model:
        Matrix4x4 M_c2m = M_m2c.inverse;

        // 5) Chain camera→world:
        Matrix4x4 M_c2w = camToWorldMatrix;
        Matrix4x4 M_m2w = M_c2w * M_m2c;


        Vector3 worldPos = M_m2w.GetColumn(3);
        Quaternion worldRot = M_m2w.rotation;

        targetTransform.position = worldPos;
        targetTransform.rotation = worldRot;
    }



    /// <summary>
    /// Build a Unity Matrix4x4 from a 9-element row-major rotation array
    /// and a translation vector.
    /// </summary>
    private static Matrix4x4 MatrixFromR_T(float[] R, Vector3 t)
    {
        var M = new Matrix4x4();
        M.m00 = R[0]; M.m01 = R[1]; M.m02 = R[2]; M.m03 = t.x;
        M.m10 = R[3]; M.m11 = R[4]; M.m12 = R[5]; M.m13 = t.y;
        M.m20 = R[6]; M.m21 = R[7]; M.m22 = R[8]; M.m23 = t.z;
        M.m30 = 0f; M.m31 = 0f; M.m32 = 0f; M.m33 = 1f;
        return M;
    }
}
