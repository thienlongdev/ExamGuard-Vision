import subprocess

cmd = 'Get-CimInstance Win32_PnPEntity | Where-Object { $_.Caption -match "camera|webcam|video|imaging|droidcam|iriun|obs" } | Select-Object Caption, PNPClass, DeviceID, Status | Format-Table -AutoSize'
p = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
print(p.stdout)
