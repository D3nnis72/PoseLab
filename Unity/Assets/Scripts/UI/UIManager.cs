using System;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public sealed class UIManager : MonoBehaviour
{
    private static UIManager _instance;
    public static UIManager Instance
    {
        get
        {
            if (_instance == null) InitializeInstance();
            return _instance;
        }
    }

    public enum View { Start, CadSelection, Settings, Scanner }

    [SerializeField] private GameObject _container;
    [SerializeField] private GameObject _startView;
    [SerializeField] private GameObject _cadView;
    [SerializeField] private GameObject _settingsView;
    [SerializeField] private GameObject _scannerView;
    [SerializeField] private TargetObjectManager _poseEstimator;
    [SerializeField] private CaptureManager _captureManager;

    private string _poseModel = string.Empty;
    private string _cadModel = string.Empty;
    private bool _isInitialized;
    private string _sceneName;
    private readonly Color _primaryColor = HexToColor("B4C6E4");
    private readonly Color _backgroundColor = HexToColor("444444");

    private static void InitializeInstance()
    {
        var existing = FindObjectOfType<UIManager>();
        if (existing != null)
            _instance = existing;
        else
        {
            var go = new GameObject("UIManager");
            _instance = go.AddComponent<UIManager>();
        }
        _instance.Initialize();
    }

    private void Awake()
    {
        if (_instance == null)
        {
            _instance = this;
            Initialize();
        }
        else if (_instance != this)
            Destroy(gameObject);
    }

    private void Initialize()
    {
        if (_isInitialized) return;
        DontDestroyOnLoad(gameObject);
        ShowView(View.Start);
        _isInitialized = true;
    }

    public void ShowStartView()
    {
        ShowView(View.Start);
    }
    public void ShowCadView()
    {
        ShowView(View.CadSelection);
    }

    public void ShowSettingsView()
    {
        ShowView(View.Settings);
    }

    public void ShowScannerView()
    {
        ShowView(View.Scanner);
    }

    public void ShowView(View view)
    {
        _container?.SetActive(true);
        _startView?.SetActive(view == View.Start);
        _cadView?.SetActive(view == View.CadSelection);
        _settingsView?.SetActive(view == View.Settings);
        _scannerView?.SetActive(view == View.Scanner);
    }

    public void HideContainer()
    {
        if (_container != null)
        {
            _container.SetActive(false);
        }
    }

    public string PoseModel
    {
        get => _poseModel;
        set
        {
            _poseModel = value ?? string.Empty;
            ToggleCadButtonColor(_poseModel);
        }

    }

    public string CadModel
    {
        get => _cadModel;
        set
        {
            _cadModel = value ?? string.Empty;
            _poseEstimator?.UpdateTargetObject(_cadModel);
        }
    }

    public string SceneName
    {
        get => _sceneName;
        set
        {
            _sceneName = value ?? string.Empty;
        }
    }

    public void ToggleCadButtonColor(string modelName)
    {
        var allButtons = GameObject.FindGameObjectsWithTag("CadModelButton");

        foreach (var obj in allButtons)
        {
            if (obj.TryGetComponent<Button>(out var btn))
            {
                String txt = btn.GetComponentInChildren<TextMeshProUGUI>().text;
                if (txt == modelName)
                {
                    SetButtonColor(btn, _primaryColor);
                }
                else
                {
                    SetButtonColor(btn, _backgroundColor);
                }
            }
        }
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
}
