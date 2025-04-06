"""
Logo generation utility for the Planning Intelligence app.
"""

import base64
from io import BytesIO


def get_logo_svg():
    """
    Generate a simple SVG logo for the Planning Intelligence app.

    Returns:
        SVG string containing the logo
    """
    # Create a simple SVG logo - blue building icon with "PI" text
    svg = f"""
    <svg width="120" height="120" viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect width="120" height="120" rx="20" fill="#1E40AF" fill-opacity="0.1"/>
        <path d="M30 30H90V90H30V30Z" fill="#1E40AF" fill-opacity="0.05"/>
        <path d="M60 20L90 40V95H30V40L60 20Z" fill="#1E40AF" fill-opacity="0.2"/>
        <path d="M40 45H50V85H40V45Z" fill="#1E40AF"/>
        <path d="M55 45H65V85H55V45Z" fill="#1E40AF"/>
        <path d="M70 45H80V85H70V45Z" fill="#1E40AF"/>
        <path d="M30 85H90V95H30V85Z" fill="#1E40AF"/>
        <path d="M60 20L90 40H30L60 20Z" fill="#1E40AF"/>
        <text x="60" y="65" font-family="Arial" font-size="30" font-weight="bold" fill="#1E40AF" text-anchor="middle">PI</text>
    </svg>
    """
    return svg


def get_logo_base64():
    """
    Get the logo as a base64 encoded string for use in HTML img tags.

    Returns:
        Base64 encoded string of the SVG logo
    """
    svg = get_logo_svg()
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"


def get_logo_html(width=80):
    """
    Get HTML img tag with the embedded logo.

    Args:
        width: Width of the logo in pixels

    Returns:
        HTML img tag with the embedded logo
    """
    b64_logo = get_logo_base64()
    return f'<img src="{b64_logo}" width="{width}px">'
