using UnityEngine;

public static class PoseUtils
{
    /// <summary>
    /// Creates a 3×3 rotation matrix around one of the principal axes.
    /// </summary>
    public static Matrix4x4 AxisAngleRotMat(char axis, float angleDeg)
    {
        float θ = Mathf.Deg2Rad * angleDeg;
        float c = Mathf.Cos(θ), s = Mathf.Sin(θ);
        switch (axis)
        {
            case 'x':
                return new Matrix4x4(
                    new Vector4(1, 0, 0, 0),
                    new Vector4(0, c, -s, 0),
                    new Vector4(0, s, c, 0),
                    new Vector4(0, 0, 0, 1)
                );
            case 'y':
                return new Matrix4x4(
                    new Vector4(c, 0, s, 0),
                    new Vector4(0, 1, 0, 0),
                    new Vector4(-s, 0, c, 0),
                    new Vector4(0, 0, 0, 1)
                );
            case 'z':
                return new Matrix4x4(
                    new Vector4(c, -s, 0, 0),
                    new Vector4(s, c, 0, 0),
                    new Vector4(0, 0, 1, 0),
                    new Vector4(0, 0, 0, 1)
                );
            default:
                throw new System.ArgumentException($"Unsupported axis '{axis}'");
        }
    }

    /// <summary>
    /// Apply the “model-space rotation about center” correction:
    ///   R_new = R0 * R_corr  
    ///   t_new = t0 - R0*c + R_new*c  
    /// where c is the mesh center (in meters or mm, as long as consistent).
    /// </summary>
    public static void ApplyModelCorrection(
        float[] R0,
        float[] t0,
        Vector3 center,
        char axis,
        float angleDeg,
        out float[] R1,
        out float[] t1
    )
    {
        // build R0 matrix
        Matrix4x4 M0 = new Matrix4x4();
        M0.SetRow(0, new Vector4(R0[0], R0[1], R0[2], 0));
        M0.SetRow(1, new Vector4(R0[3], R0[4], R0[5], 0));
        M0.SetRow(2, new Vector4(R0[6], R0[7], R0[8], 0));
        M0.m33 = 1;

        // correction
        Matrix4x4 Rcorr = AxisAngleRotMat(axis, angleDeg);
        Matrix4x4 M1 = M0 * Rcorr;

        // translate
        Vector3 t0v = new Vector3(t0[0], t0[1], t0[2]);
        Vector3 tNew = t0v - M0.MultiplyPoint3x4(center) + M1.MultiplyPoint3x4(center);

        // flatten back
        R1 = new float[9] {
            M1.m00, M1.m01, M1.m02,
            M1.m10, M1.m11, M1.m12,
            M1.m20, M1.m21, M1.m22
        };
        t1 = new float[3] { tNew.x, tNew.y, tNew.z };
    }

    /// <summary>
    /// Reflect the pose about a model-space axis (through the given center point):
    ///   R_new = R0 * D  
    ///   t_new = t0 + R0*(I – D)*c  
    /// where D = diag(±1,±1,±1) for the chosen axis.
    /// </summary>
    public static void MirrorModelAxisCenter(
        float[] R0,
        float[] t0,
        Vector3 center,
        char axis,
        out float[] Rm,
        out float[] tm
    )
    {
        // unpack R0
        Matrix4x4 M0 = new Matrix4x4();
        M0.SetRow(0, new Vector4(R0[0], R0[1], R0[2], 0));
        M0.SetRow(1, new Vector4(R0[3], R0[4], R0[5], 0));
        M0.SetRow(2, new Vector4(R0[6], R0[7], R0[8], 0));
        M0.m33 = 1;

        // build D
        Vector3 d = Vector3.one;
        switch (axis)
        {
            case 'x': d.x = -1; break;
            case 'y': d.y = -1; break;
            case 'z': d.z = -1; break;
            default: throw new System.ArgumentException($"Unsupported axis '{axis}'");
        }
        Matrix4x4 D = Matrix4x4.Scale(d);

        Matrix4x4 Mref = M0 * D;


        Vector3 t0v = new Vector3(t0[0], t0[1], t0[2]);
        Vector3 diff = Vector3.Scale(Vector3.one - d, center);
        Vector3 tRef = t0v + M0.MultiplyVector(diff);


        Rm = new float[9] {
            Mref.m00, Mref.m01, Mref.m02,
            Mref.m10, Mref.m11, Mref.m12,
            Mref.m20, Mref.m21, Mref.m22
        };
        tm = new float[3] { tRef.x, tRef.y, tRef.z };
    }

    /// <summary>
    /// Inverse of ApplyModelCorrection:
    /// Given R1 = R0·Rcorr, t1 = t0 - R0·c + R1·c,
    /// recover R0, t0.
    /// </summary>
    public static void ApplyModelCorrectionInverse(
        float[] R1,
        float[] t1,
        Vector3 center,
        char axis,
        float angleDeg,
        out float[] R0,
        out float[] t0
    )
    {

        Matrix4x4 M1 = new Matrix4x4();
        M1.SetRow(0, new Vector4(R1[0], R1[1], R1[2], 0));
        M1.SetRow(1, new Vector4(R1[3], R1[4], R1[5], 0));
        M1.SetRow(2, new Vector4(R1[6], R1[7], R1[8], 0));
        M1.m33 = 1;

        Matrix4x4 Rcorr = AxisAngleRotMat(axis, angleDeg);
        Matrix4x4 M0 = M1 * Rcorr.transpose;

        Vector3 t1v = new Vector3(t1[0], t1[1], t1[2]);
        Vector3 Rc = M0.MultiplyPoint3x4(center);
        Vector3 R1c = M1.MultiplyPoint3x4(center);
        Vector3 t0v = t1v + Rc - R1c;

        R0 = new float[9] {
            M0.m00, M0.m01, M0.m02,
            M0.m10, M0.m11, M0.m12,
            M0.m20, M0.m21, M0.m22
        };
        t0 = new float[3] { t0v.x, t0v.y, t0v.z };
    }

    /// <summary>
    /// Inverse of MirrorModelAxisCenter:
    /// Given Rm = R0·D, t_ref = t0 + R0·(I–D)·c,
    /// recover R0, t0.
    /// </summary>
    public static void MirrorModelAxisCenterInverse(
        float[] Rm,
        float[] t_ref,
        Vector3 center,
        char axis,
        out float[] R0,
        out float[] t0
    )
    {
        Matrix4x4 Mref = new Matrix4x4();
        Mref.SetRow(0, new Vector4(Rm[0], Rm[1], Rm[2], 0));
        Mref.SetRow(1, new Vector4(Rm[3], Rm[4], Rm[5], 0));
        Mref.SetRow(2, new Vector4(Rm[6], Rm[7], Rm[8], 0));
        Mref.m33 = 1;

        Vector3 d = Vector3.one;
        switch (axis)
        {
            case 'x': d.x = -1; break;
            case 'y': d.y = -1; break;
            case 'z': d.z = -1; break;
            default: throw new System.ArgumentException($"Unsupported axis '{axis}'");
        }
        Matrix4x4 D = Matrix4x4.Scale(d);

        Matrix4x4 M0 = Mref * D;

        Vector3 trefv = new Vector3(t_ref[0], t_ref[1], t_ref[2]);
        Vector3 diff = Vector3.Scale(Vector3.one - d, center);
        Vector3 t0v = trefv - M0.MultiplyVector(diff);

        R0 = new float[9] {
            M0.m00, M0.m01, M0.m02,
            M0.m10, M0.m11, M0.m12,
            M0.m20, M0.m21, M0.m22
        };
        t0 = new float[3] { t0v.x, t0v.y, t0v.z };
    }
}
