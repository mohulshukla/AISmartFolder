import subprocess
import time
from typing import Optional


def send_notification(
    title: str, message: str, subtitle: Optional[str] = None, sound: bool = True
) -> None:
    """
    Send a beautiful macOS notification using osascript

    Args:
        title: The notification title
        message: The notification message
        subtitle: Optional subtitle for the notification
        sound: Whether to play the default notification sound
    """
    # Escape double quotes in the strings
    title = title.replace('"', '\\"')
    message = message.replace('"', '\\"')
    subtitle = subtitle.replace('"', '\\"') if subtitle else ""

    script = f'''
    display notification "{message}" with title "{title}"'''

    if subtitle:
        script += f' subtitle "{subtitle}"'

    if sound:
        script += ' sound name "default"'

    subprocess.run(["osascript", "-e", script])


def demo_notifications():
    # Simple notification
    send_notification("Hello! 👋", "Welcome to the notification demo!")

    # Wait 2 seconds
    time.sleep(2)

    # Notification with subtitle
    send_notification(
        "🎉 Project Update",
        "Everything is running smoothly. 🚀",
        subtitle="Deployment Status",
    )

    time.sleep(2)

    # Tip notification
    send_notification(
        "💡 Quick Tip",
        "Press ⌘+Space to open Spotlight Search",
        subtitle="Keyboard Shortcuts",
        sound=False,  # Silent notification
    )


if __name__ == "__main__":
    demo_notifications()
