# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\heet1\\PyCharmMiscProject\\face_attendance_webapp\\.venv\\Lib\\site-packages\\face_recognition', 'face_recognition'), ('C:\\Users\\heet1\\PyCharmMiscProject\\face_attendance_webapp\\.venv\\Lib\\site-packages\\face_recognition_models\\models', 'face_recognition_models/models'), ('C:\\Users\\heet1\\PyCharmMiscProject\\face_attendance_webapp\\templates', 'templates'), ('C:\\Users\\heet1\\PyCharmMiscProject\\face_attendance_webapp\\static', 'static')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
