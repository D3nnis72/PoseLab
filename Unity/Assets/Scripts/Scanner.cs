using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;

public class Scanner : MonoBehaviour
{
    [SerializeField] private TargetObjectManager poseEstimator;

    [Header("Lock/Unlock Buttons (1→6)")]
    [SerializeField] private Button btn1;
    [SerializeField] private Button btn2;
    [SerializeField] private Button btn3;
    [SerializeField] private Button btn4;
    [SerializeField] private Button btn5;
    [SerializeField] private Button btn6;


    private Dictionary<string, bool> _markerLocks = new Dictionary<string, bool>()
    {
        { "1", true },
        { "2", true },
        { "3", true },
        { "4", true },
        { "5", true },
        { "6", true },
    };

    private Dictionary<string, Button> _buttonMap;

    private readonly Color _unlockedColor = HexToColor("B4C6E4");
    private readonly Color _lockedColor = HexToColor("444444");

    void Awake()
    {
        if (poseEstimator == null)
            Debug.LogError("Scanner: missing TargetObjectManager ref!");

        // map marker IDs to buttons
        _buttonMap = new Dictionary<string, Button>
        {
            { "1", btn1 },
            { "2", btn2 },
            { "3", btn3 },
            { "4", btn4 },
            { "5", btn5 },
            { "6", btn6 },
        };

        // wire up onClick listeners
        btn1.onClick.AddListener(() => ToggleLock("1"));
        btn2.onClick.AddListener(() => ToggleLock("2"));
        btn3.onClick.AddListener(() => ToggleLock("3"));
        btn4.onClick.AddListener(() => ToggleLock("4"));
        btn5.onClick.AddListener(() => ToggleLock("5"));
        btn6.onClick.AddListener(() => ToggleLock("6"));

        // initialize button colors
        foreach (var kv in _buttonMap)
            SetButtonColor(kv.Key, _lockedColor);
    }

    private void ToggleLock(string markerName)
    {
        if (!_markerLocks.ContainsKey(markerName)) return;

        // flip lock state
        bool locked = !_markerLocks[markerName];
        _markerLocks[markerName] = locked;

        // update button BG color
        SetButtonColor(markerName, locked ? _lockedColor : _unlockedColor);

        Debug.Log($"Marker {markerName} is now {(locked ? "LOCKED" : "UNLOCKED")}");
    }

    private void SetButtonColor(string markerName, Color col)
    {
        if (_buttonMap.TryGetValue(markerName, out var btn))
        {
            var img = btn.GetComponent<Image>();
            if (img != null) img.color = col;
        }
    }

    private static Color HexToColor(string hex)
    {
        if (ColorUtility.TryParseHtmlString("#" + hex, out var c))
            return c;
        return Color.white;
    }


    void OnEnable() => poseEstimator.trackedImageManager.trackablesChanged.AddListener(OnChanged);
    void OnDisable() => poseEstimator.trackedImageManager.trackablesChanged.RemoveListener(OnChanged);

    private void OnChanged(ARTrackablesChangedEventArgs<ARTrackedImage> args)
    {
        foreach (var added in args.added)
            SpawnMarker(added);

        foreach (var updated in args.updated)
            TryUpdateMarker(updated);

        foreach (var removed in args.removed)
            RemoveMarker(removed.Value);

        poseEstimator.UpdateTargetObjectPose();
    }

    private void SpawnMarker(ARTrackedImage trackedImage)
    {
        var name = trackedImage.referenceImage.name;

        if (_markerLocks.TryGetValue(name, out bool isLocked) && isLocked)
            return;

        poseEstimator.SpawnMarker(trackedImage);
    }

    private void TryUpdateMarker(ARTrackedImage trackedImage)
    {
        var name = trackedImage.referenceImage.name;

        if (_markerLocks.TryGetValue(name, out bool isLocked) && isLocked)
            return;

        poseEstimator.UpdateMarker(trackedImage);
    }

    private void RemoveMarker(ARTrackedImage trackedImage)
    {
        var name = trackedImage.referenceImage.name;

        if (_markerLocks.TryGetValue(name, out bool isLocked) && isLocked)
            return;

        poseEstimator.RemoveMarker(trackedImage);
    }
}
