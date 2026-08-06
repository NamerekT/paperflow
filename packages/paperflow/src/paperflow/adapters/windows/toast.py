"""A native Windows 11 notification, with no extra library and no AUMID registration.

Shelling out to PowerShell and building the toast XML there is what avoids needing an
app registered with the shell just to show one notification. `open_target`, if given (a
folder path or an http(s) URL), makes clicking the notification open it - Explorer for a
folder, the default browser for a URL - through protocol activation, which again needs
no registration.

Title and body are handed to PowerShell as base64-encoded UTF-16LE rather than
interpolated into the script text directly: that is what makes quotes, Cyrillic,
umlauts and newlines in a document title safe to pass through without turning this into
a string-escaping exercise inside a `-Command` argument.
"""

import base64
import logging
import subprocess

logger = logging.getLogger(__name__)

#: The toast's notifier id - what Windows groups notifications under. Renamed from the
#: old DocSorter alongside the rest of the project; has no effect on anything already
#: shown, only on notifications from this point on.
APP_ID = "Paperflow"


def _ps_b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-16-le")).decode("ascii")


def send_toast(title: str, body: str, *, open_target: str | None = None) -> None:
    """Show a toast. Never raises - a failed notification must not fail whatever
    triggered it; the caller has already done the thing worth notifying about."""
    title_b64 = _ps_b64(title)
    body_b64 = _ps_b64(body)
    target_b64 = _ps_b64(open_target) if open_target else ""

    script = f"""
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType = WindowsRuntime] | Out-Null
[Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null

$title  = [System.Text.Encoding]::Unicode.GetString([System.Convert]::FromBase64String('{title_b64}'))
$body   = [System.Text.Encoding]::Unicode.GetString([System.Convert]::FromBase64String('{body_b64}'))
$titleX = [System.Security.SecurityElement]::Escape($title)
$bodyX  = [System.Security.SecurityElement]::Escape($body)

$launchAttr = ''
$targetB64  = '{target_b64}'
if ($targetB64 -ne '') {{
    $target = [System.Text.Encoding]::Unicode.GetString([System.Convert]::FromBase64String($targetB64))
    $uriX   = [System.Security.SecurityElement]::Escape(([Uri]$target).AbsoluteUri)
    $launchAttr = " activationType=`"protocol`" launch=`"$uriX`""
}}

$toastXmlString = "<toast$launchAttr><visual><binding template=`"ToastGeneric`"><text>$titleX</text><text>$bodyX</text></binding></visual></toast>"

$xmlDoc = New-Object Windows.Data.Xml.Dom.XmlDocument
$xmlDoc.LoadXml($toastXmlString)

$toast = [Windows.UI.Notifications.ToastNotification]::new($xmlDoc)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{APP_ID}').Show($toast)
"""
    try:
        subprocess.Popen(
            ["powershell", "-WindowStyle", "Hidden", "-NonInteractive", "-Command", script],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except Exception:
        logger.debug("toast notification failed", exc_info=True)
